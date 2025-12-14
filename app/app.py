from fastapi import FastAPI
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI()

# ChromaDB クライアント
client = chromadb.Client()
collection = client.get_or_create_collection("docs")

# SentenceTransformer モデル
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.api_route("/ingest", methods=["GET", "POST"])
def ingest():
    docs = [
        "エレン・イェーガーは、人類を脅かす巨人に強い憎しみを抱く物語の主人公である。",
        "人類は巨大な壁に囲まれた都市で生活し、外の世界には巨人が存在している。",
        "調査兵団は壁の外に出て巨人の調査や討伐を行う組織である。",
        "ミカサ・アッカーマンはエレンの幼なじみで、非常に高い戦闘能力を持つ。",
        "巨人には人間が変身してなる知性を持つ存在と、意思を持たない無垢の巨人がいる。"
    ]

    embeddings = model.encode(docs).tolist()

    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=[f"id_{i}" for i in range(len(docs))]
    )

    return {"message": "ドキュメントを追加しました", "count": len(docs)}


@app.get("/search")
def search(query: str, top_k: int = 1):
    q_emb = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k
    )
    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]

    n = min(top_k, len(docs))
    hits = [
        {"id": ids[i], "document": docs[i], "distance": distances[i]}
        for i in range(n)
    ]

    return {"query": query, "hits": hits}