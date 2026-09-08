%%writefile app.py
import streamlit as st
import os

from embeddings import load_embedding_model
from vector_store import process_and_index_file
from rag_chain import query_rag_pipeline

st.set_page_config(page_title="RAG PDF QA", layout="wide")
st.title("📄 RAG App (Groq + FAISS)")

groq_api_key = os.environ.get("GROQ_API_KEY", "")

with st.sidebar:
    st.header("Settings")
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password")
    
    selected_model = st.selectbox(
        "Select Model",
        options=["openai/gpt-oss-120b", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    )

@st.cache_resource
def get_embeddings():
    return load_embedding_model()

embeddings = get_embeddings()

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    if "vectorstore" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Processing document and generating vector index..."):
            try:
                vectorstore, num_chunks = process_and_index_file(uploaded_file, embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.file_name = uploaded_file.name
                st.success(f"Indexed successfully into {num_chunks} chunks!")
            except Exception as e:
                st.error(f"Error: {e}")

if "vectorstore" in st.session_state:
    st.divider()
    user_query = st.text_input("Ask a question about your document:")

    if user_query:
        if not groq_api_key:
            st.error("Missing Groq API Key!")
        else:
            with st.spinner("Thinking..."):
                answer, source_docs = query_rag_pipeline(
                    query=user_query,
                    vectorstore=st.session_state.vectorstore,
                    groq_api_key=groq_api_key,
                    model_name=selected_model
                )
                st.subheader("Answer:")
                st.write(answer)

                with st.expander("View Retrieved Chunks"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.text(doc.page_content)
