from langchain_huggingface import HuggingFaceEmbeddings

def load_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
