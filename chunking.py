import re

def split_text(text, chunk_size=300, overlap=50):
    """
    Splits text into smaller chunks (approx 60-75 tokens) to fit FLAN-T5 context.
    Strictly respecting sentence boundaries where possible.
    """
    if not text:
        return []

    # Clean up newlines that might break sentences in PDFs
    text = text.replace('.\n', '. ').replace('\n', ' ')
    
    # Split by sentence terminators
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Create a new chunk, checking if the single sentence itself is huge
            if len(sentence) > chunk_size:
                # Force split long sentences
                start = 0
                while start < len(sentence):
                    end = start + chunk_size
                    chunks.append(sentence[start:end])
                    start += chunk_size - overlap
                current_chunk = "" 
            else:
                current_chunk = sentence + " "
                
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
