from fastapi import FastAPI
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI()
client = chromadb.Client()
collection = client.get_or_create_collection("docs")
model = SentenceTransformer("all-MiniLm-L6-v2")

@app.get("/search")
def search(query: str):
    q_emb = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb, n_results=3
    )
    return {"query": query, "results": results}
