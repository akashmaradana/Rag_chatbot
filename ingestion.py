import os
import fitz  # PyMuPDF

def load_document(file_path):
    """
    Loads text from a PDF or TXT file using PyMuPDF for better quality.
    Args:
        file_path (str): Path to the file.
    Returns:
        str: Extracted text.
    """
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    try:
        if ext == ".pdf":
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
        # Basic cleanup
        text = text.replace('\xa0', ' ').replace('  ', ' ')
        return text.strip()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return ""
