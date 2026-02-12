# 📚 Precision RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot built completely with **open-source** models. Designed for accuracy, privacy, and performance without any paid APIs.

## 🚀 Features
- **Pure Open Source**: Uses `flan-t5-base` (LLM) and `all-MiniLM-L6-v2` (Embeddings).
- **Zero Hallucination Focus**: Strict prompt engineering and deterministic generation.
- **Efficient Retrieval**: utilizing FAISS for fast vector similarity search.
- **Smart Chunking**: Text processing optimized for the model's token limits.
- **User-Friendly UI**: Clean, responsive interface built with Streamlit.

## 🛠️ Tech Stack
- **Language**: Python 3.8+
- **Frontend**: Streamlit
- **LLM**: `google/flan-t5-base`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB**: FAISS
- **PDF Processing**: PyMuPDF

## 📂 Project Structure
```
Rag/
├── app.py              # Main application entry point (UI)
├── generation.py       # LLM generation logic (FLAN-T5)
├── retrieval.py        # Vector store and embedding logic (FAISS)
├── ingestion.py        # Document loading (PDF/TXT)
├── chunking.py         # Text splitting logic
├── requirements.txt    # Project dependencies
├── test_generation.py  # Verification script
├── data/               # Directory for storing uploaded docs
└── embeddings/         # Directory for saving FAISS index
```

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

### 3. Usage
1.  Open the link provided in the terminal (usually `http://localhost:8501`).
2.  Use the sidebar to **Upload** a PDF or TXT file.
3.  Click **Process & Index**.
4.  Once ready, type your question in the chat box!

## 🧠 Model Constraints & Optimization
- **Token Limit**: `flan-t5-base` has a strict 512-token limit.
- **Chunk Size**: Documents are chunked into small segments (~300 chars) to ensure multiple relevant contexts fit within the model's window.
- **Generation**: We use `do_sample=False` (greedy decoding) to strictly adhere to the retrieved context and prevent "creative" hallucinations.

---
*Built with ❤️ using Open Source AI.*
