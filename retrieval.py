import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Paths
INDEX_PATH = "embeddings/faiss_index/index.faiss"
METADATA_PATH = "embeddings/faiss_index/metadata.pkl"

def get_embedding_model():
    """
    Loads and returns the SentenceTransformer model.
    Designed to be cached by the calling application.
    """
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class VectorStore:
    def __init__(self, embedding_model=None):
        if embedding_model:
            self.model = embedding_model
        else:
            self.model = get_embedding_model()
            
        self.dimension = 384
        self.index = None
        self.metadata = []
        self.load_index()

    def load_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            try:
                self.index = faiss.read_index(INDEX_PATH)
                with open(METADATA_PATH, "rb") as f:
                    self.metadata = pickle.load(f)
            except Exception as e:
                print(f"Error loading index: {e}, creating new one.")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.metadata = []
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []

    def save_index(self):
        if not os.path.exists("embeddings/faiss_index"):
            os.makedirs("embeddings/faiss_index")
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

    def add_texts(self, texts):
        if not texts:
            return
        
        # Batch encode for speed
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        if self.index is None:
             self.index = faiss.IndexFlatL2(self.dimension)
             
        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(texts)
        self.save_index()

    def search(self, query, k=5):
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_embedding.astype("float32"), k)
        
        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results


