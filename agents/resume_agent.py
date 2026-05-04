class ResumeAgent:

    def __init__(self, vectordb, embedder):
        self.db = vectordb
        self.embedder = embedder

    def run(self, state: dict):

        resume_text = state["resume_text"]

        if len(resume_text.strip()) < 100:
        state["resume_warning"] = "이력서 내용이 부족하여 정확한 분석이 어려울 수 있습니다."

        chunks = [
            resume_text[i:i+500]
            for i in range(0, len(resume_text), 500)
        ]

        embeddings = self.embedder.embed(chunks)

        self.db.add(embeddings, chunks)

        query = "주요 경력과 기술 요약"

        q_emb = self.embedder.embed([query])

        retrieved = self.db.search(q_emb)

        state["resume_context"] = "\n".join(retrieved)

        return state
