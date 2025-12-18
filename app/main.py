import os
from fastapi import FastAPI
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# setup ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection("docs")

# setup SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
load_dotenv()

# setup LLM
LLM_MODEL = "gpt-4.1-nano"
api_key = os.getenv("OPENAI_API_KEY")
client_llm = OpenAI(api_key=api_key)

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

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

def generate_answer(query: str, contexts: list[str]):
    context_text = "\n".join(contexts)

    prompt = f"""
    以下は参考情報です。
    ---
    {context_text}
    ---

    この情報をもとに、次の質問に日本語で答えてください。

    質問: {query}
    """
    try: 
        response = client_llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    except Exception as e:
        return f"LLM呼び出しに失敗しました: {e}"
    
    return response.choices[0].message.content

@app.post("/ask", response_class=HTMLResponse)
def ask_html(query: str):
    q_emb = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=3
    )

    docs = results["documents"][0]
    answer = generate_answer(query, docs)

    return f"""
    <h3>回答</h3>
    <p>{answer}</p>
    """