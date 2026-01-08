from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Virat Kohli is known for his aggressive batting style and consistency across all formats of cricket.",
    "Sachin Tendulkar is regarded as one of the greatest cricketers of all time and is called the Master Blaster.",
    "MS Dhoni is famous for his calm leadership and led India to multiple ICC trophies.",
    "Rohit Sharma is known for his elegant batting and holds the record for the highest individual score in ODI cricket.",
    "Jasprit Bumrah is recognized for his unorthodox bowling action and deadly yorkers."
]

query = "Who is the best keeper"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)