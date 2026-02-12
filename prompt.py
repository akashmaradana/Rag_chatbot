"""
Prompt construction utilities.

We build a strict instruction that:
- Tells the model to use ONLY the provided context.
- Forces it to answer "Information not found." if the answer
  cannot be derived directly from that context.
"""

from typing import List

from retrieval import RetrievedChunk


NOT_FOUND_MESSAGE = "Information not found."


def build_prompt(question: str, contexts: List[RetrievedChunk]) -> str:
    """
    Construct a prompt for FLAN-T5.

    The prompt structure explicitly instructs the model to:
    - Rely only on the given context.
    - Avoid adding outside knowledge.
    - Reply with a fixed fallback string when the answer is missing.
    """
    context_blocks = []
    for i, c in enumerate(contexts, start=1):
        block = f"Source {i} (file: {c.source}):\n{c.text}"
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    # Keep prompt short to reduce latency and hallucinations.
    prompt = f"""Answer the question using ONLY the context.
If the answer is not fully in the context, reply exactly: {NOT_FOUND_MESSAGE}

Context:
{context_text}

Question: {question.strip()}
Answer:""".strip()

    return prompt

