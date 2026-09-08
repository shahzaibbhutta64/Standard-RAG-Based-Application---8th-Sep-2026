from langchain_huggingface import HuggingFaceEmbeddings

def load_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # Force CPU execution to prevent GPU lookup failures on Streamlit Cloud
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings
