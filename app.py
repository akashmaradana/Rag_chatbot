import streamlit as st
import os
import ingestion
import chunking
import retrieval
import generation

st.set_page_config(page_title="Local RAG Chatbot", layout="wide")

st.title("📚 Precision RAG Chatbot")
st.markdown("Optimized for accuracy with `flan-t5-base`.")

# --- Caching for Performance ---
@st.cache_resource
def load_llm():
    return generation.get_llm_pipeline()

@st.cache_resource
def load_embedding_model():
    return retrieval.get_embedding_model()

# Load models at startup (cached)
with st.spinner("Loading AI Models..."):
    llm_pipeline = load_llm()
    embedding_model = load_embedding_model()

# Initialize Vector Store with cached model
if "vector_store" not in st.session_state:
    st.session_state.vector_store = retrieval.VectorStore(embedding_model)

# Sidebar
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
    
    if uploaded_file:
        # Check if we already processed this file
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.status("Processing document...", expanded=True) as status:
                # Clear old index
                st.session_state.vector_store = retrieval.VectorStore(embedding_model)
                status.write("Cleared previous index.")

                # Save
                file_path = os.path.join("data/documents", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                status.write("File saved.")
                
                # Ingest
                try:
                    text = ingestion.load_document(file_path)
                    status.write(f"Extracted {len(text)} characters.")
                    
                    # Chunk (Strict 300 chars)
                    chunks = chunking.split_text(text, chunk_size=300, overlap=50)
                    status.write(f"Split into {len(chunks)} chunks.")
                    
                    # Embed
                    st.session_state.vector_store.add_texts(chunks)
                    
                    # Mark as processed
                    st.session_state.processed_file = uploaded_file.name
                    
                    status.update(label="Processing Complete!", state="complete", expanded=False)
                    st.success(f"Ready! ({len(chunks)} chunks)")
                except Exception as e:
                    status.update(label="Error!", state="error")
                    st.error(f"Failed: {e}")
        else:
            # Already processed
             st.success(f"Document loaded: {uploaded_file.name}")

    st.divider()
    st.checkbox("Debug Mode (Show Prompt)", key="debug_mode")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Retrieve
        relevant_docs = st.session_state.vector_store.search(prompt, k=4)
        
        if not relevant_docs:
            response = "Information not found in the documents."
            context = ""
        else:
            context = "\n\n".join(relevant_docs)
            response = generation.generate_answer(llm_pipeline, context, prompt)
        
        message_placeholder.markdown(response)
        
        # Debug View
        if st.session_state.debug_mode:
            with st.expander("🛠️ Debug Information", expanded=True):
                st.subheader("Retrieved Context Chunks")
                for i, doc in enumerate(relevant_docs):
                    st.text(f"Chunk {i+1} ({len(doc)} chars):\n{doc}")
                
                st.subheader("Full Context String")
                st.code(context)

    st.session_state.messages.append({"role": "assistant", "content": response})
