from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

model_id = "google/flan-t5-base"

def get_llm_pipeline():
    """
    Loads and returns the FLAN-T5 pipeline. 
    Designed to be cached by the calling application.
    """
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Check for GPU
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    return pipe

def generate_answer(pipeline_obj, context, question):
    """
    Generates an answer using the provided pipeline object.
    With strict token limiting to prevent context truncation.
    """
    
    # Constants for FLAN-T5-Base
    MAX_SEQ_LEN = 512
    MAX_NEW_TOKENS = 100
    
    # Construct the prompt template to measure its size
    # We use a placeholder for context to calculate overhead
    prompt_template = "Context: {}\n\nQuestion: {}\n\nAnswer:"
    
    # Calculate tokens for the fixed parts (template + question)
    tokenizer = pipeline_obj.tokenizer
    
    # Crude approximation of template overhead + question tokens
    # To be safe, we'll reserve space.
    # A more precise way would be to tokenize the template without context, but this is sufficient.
    question_tokens = len(tokenizer.encode(question))
    template_overhead = 30 # Rough estimate for "Context: ... Question: ... Answer:"
    
    available_tokens_for_context = MAX_SEQ_LEN - MAX_NEW_TOKENS - question_tokens - template_overhead
    
    if available_tokens_for_context < 0:
        # Question is too long, we need to truncate it or handle error
        # For now, let's just use a minimal context window
        available_tokens_for_context = 50 
    
    # Tokenize context and truncate
    context_tokens = tokenizer.encode(context)
    if len(context_tokens) > available_tokens_for_context:
        # Truncate
        context_tokens = context_tokens[:available_tokens_for_context]
        context = tokenizer.decode(context_tokens, skip_special_tokens=True)
    
    # Hybrid RAG Prompt
    prompt = (
        f"Context provided:\n{context}\n\n"
        f"Instruction: You are a production-grade AI. Answer the question using the following RULES:\n"
        f"1. Use your own knowledge first. Use the context ONLY if it is relevant.\n"
        f"2. If the context is irrelevant, ignore it.\n"
        f"3. Do NOT hallucinate. Note uncertainty if unsure.\n"
        f"4. Format your answer exactly as follows:\n\n"
        f"Direct Answer: [Your direct answer]\n"
        f"Reasoning: [Explain using your general knowledge]\n"
        f"Evidence: [Summarize relevant context points, or say 'None' if context is irrelevant]\n"
        f"Confidence: [High/Medium/Low]\n\n"
        f"Question: {question}\n\n"
        f"Response:"
    )
    
    # Run inference
    result = pipeline_obj(
        prompt, 
        max_new_tokens=300,           # Increased for structured output
        do_sample=False,              # Deterministic for instruction following
    )
    return result[0]['generated_text']
