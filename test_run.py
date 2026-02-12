from agents.resume_agent import ResumeAgent
from agents.job_agent import JobAgent
from agents.skill_agent import SkillAgent
from agents.coach_agent import CoachAgent

from rag.embedder import Embedder
from rag.vector_db import VectorDB
from graph.workflow import build_graph


resume_text = """
Python 개발자, 3년 경력
FastAPI, Docker 경험
AI 프로젝트 수행
"""

job_text = """
We are looking for backend developer.
Python, AWS, MLOps preferred.
3+ years experience.
"""


embedder = Embedder()
vectordb = VectorDB()

resume_agent = ResumeAgent(vectordb, embedder)
job_agent = JobAgent()
skill_agent = SkillAgent()
coach_agent = CoachAgent()

graph = build_graph(
    resume_agent,
    job_agent,
    skill_agent,
    coach_agent
)

state = {
    "resume_text": resume_text,
    "job_text": job_text
}

result = graph.invoke(state)

print("\n=== FINAL RESULT ===\n")
print(result["final_advice"])
