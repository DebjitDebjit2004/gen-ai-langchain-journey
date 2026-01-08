from langchain_huggingface import HuggingFaceEmbeddings

# Local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Sample documents
documents = [
    "India has one of the fastest growing economies.",
    "LangChain is a framework for building LLM-powered applications.",
    "Machine learning enables computers to learn from data."
]

# Create embeddings
vectors = embeddings.embed_documents(documents)

print(vectors)
