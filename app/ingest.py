import chromadb
from sentence_transformers import SentenceTransformer
import os

def load_docs(path="data/docs"):
    """RAG用のテキストデータを読み込み"""
    docs = []
    for filename in os.listdir(path):
        open_file = os.path.join(path, filename)
        with open(open_file, "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs

def main():
    #Hugging FaceのSentence Transformersプロジェクトで提供されている軽量かつ高速な文章埋め込みモデル
    model = SentenceTransformer("all-MiniLM-L6-v2") 
    client = chromadb.Client()
    collection = client.get_or_create_collection("docs")

    # embedding
    docs = load_docs()
    embeddings = model.encode(docs).tolist() #embeddingしたものをリスト化

    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=[f"docs-{i}" for i in range(len(docs))]
    )

    print("Documents ingested")

if __name__=="__main__":
    main()