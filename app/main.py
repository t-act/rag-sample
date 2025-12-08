from fastapi import FastAPI
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI()

# ChromaDB クライアント
client = chromadb.Client()
collection = client.get_or_create_collection("docs")

# SentenceTransformer モデル
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/ingest")
def ingest():
    docs = [
        "太陽は非常に熱い天体で、表面温度は約5500度です。",
        "この物語の主人公はアリスという少女です。",
        "富士山は日本で最も高い山です。"
    ]

    embeddings = model.encode(docs).tolist()

    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=[f"id_{i}" for i in range(len(docs))]
    )

    return {"message": "ドキュメントを追加しました", "count": len(docs)}


@app.get("/search")
def search(query: str):
    q_emb = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=3
    )
    return {"query": query, "results": results}