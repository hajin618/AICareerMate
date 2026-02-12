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
