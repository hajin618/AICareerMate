import faiss
import numpy as np

class VectorDB:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings))
        self.texts.extend(texts)

    def search(self, query_emb, k=3):
        D, I = self.index.search(query_emb, k)
        return [self.texts[i] for i in I[0]]


# 전처리 함수
def preprocess_text(text: str):
    lines = text.split("\n")

    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) > 5:
            cleaned.append(line)

    return list(set(cleaned))


# RAG 검색 함수
def retrieve_context(vector_db, query_emb, k=3):
    return vector_db.search(query_emb, k)