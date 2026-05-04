import os
import re
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rag.loader import load_pdf, preprocess_text
from rag.embedder import Embedder
from rag.vector_db import VectorDB

from agents.resume_agent import ResumeAgent
from agents.job_agent import JobAgent
from agents.skill_agent import SkillAgent
from agents.coach_agent import CoachAgent
from graph.workflow import build_graph

from schemas import AnalysisResponse

app = FastAPI(title="AI CareerMate API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 유틸 함수
def safe_int_from_text(text: str, default: int = 60) -> int:
    match = re.search(r"적합도\s*점수\s*:\s*-?\s*(\d{1,3})", text)
    if match:
        value = int(match.group(1))
        return max(0, min(100, value))
    return default


def extract_section_list(text: str, section_name: str) -> list[str]:
    pattern = rf"{section_name}\s*:\s*((?:\n- .+)+)"
    match = re.search(pattern, text)
    if not match:
        return []

    block = match.group(1)
    return [
        line.replace("- ", "").strip()
        for line in block.splitlines()
        if line.strip().startswith("- ")
    ]


# API 엔드포인트
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(...),
    job_text: str = Form(...)
):
    
    # 입력 검증
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    if not job_text.strip():
        raise HTTPException(status_code=400, detail="채용공고(job_text)는 필수입니다.")

    safe_name = f"{uuid.uuid4().hex}.pdf"
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        
        # 파일 저장
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        
        # 1. PDF 로드
        resume_text = load_pdf(file_path)

        if not resume_text or len(resume_text.strip()) < 30:
            raise HTTPException(status_code=400, detail="이력서 내용이 너무 짧거나 추출에 실패했습니다.")

        if len(job_text.strip()) < 20:
            raise HTTPException(status_code=400, detail="채용공고 내용이 너무 짧습니다.")

        
        # 2. 전처리
        chunks = preprocess_text(resume_text)

        
        # 3. 임베딩 생성
        embedder = Embedder()
        embeddings = embedder.embed(chunks)

        
        # 4. Vector DB 구성
        vector_db = VectorDB(dim=embeddings.shape[1])
        vector_db.add(embeddings, chunks)

        
        # 5. RAG 검색
        query_emb = embedder.embed([job_text])
        retrieved_chunks = vector_db.search(query_emb, k=3)

        # context 생성
        resume_context = "\n".join(retrieved_chunks)

        # 6. Agent 생성
        resume_agent = ResumeAgent(vector_db, embedder)
        job_agent = JobAgent()
        skill_agent = SkillAgent()
        coach_agent = CoachAgent()

        
        # 7. LangGraph 구성
        graph = build_graph(
            resume_agent=resume_agent,
            job_agent=job_agent,
            skill_agent=skill_agent,
            coach_agent=coach_agent
        )

        
        # 8. 상태(state) 구성
        state = {
            "resume_text": resume_text,
            "job_text": job_text,
            "resume_context": resume_context,
            "retrieved_chunks": retrieved_chunks
        }

        
        # 9. 그래프 실행
        result = graph.invoke(state)

        
        # 10. 결과 후처리
        skill_analysis = result.get("skill_analysis", "")
        final_advice = result.get("final_advice", "")

        fit_score = safe_int_from_text(skill_analysis, default=60)
        strengths = extract_section_list(skill_analysis, "강점")
        gaps = extract_section_list(skill_analysis, "부족 역량")
        recommended_actions = extract_section_list(skill_analysis, "추천 액션")

        # fallback 처리
        if not strengths:
            strengths = ["이력서 기반 일부 기술 역량이 직무와 부합합니다."]
        if not gaps:
            gaps = ["채용공고 대비 보완이 필요한 역량이 존재합니다."]
        if not recommended_actions:
            recommended_actions = ["부족 기술 중심으로 학습 계획 수립을 권장합니다."]

        
        # 11. 응답 반환
        return AnalysisResponse(
            fit_score=fit_score,
            strengths=strengths,
            gaps=gaps,
            recommended_actions=recommended_actions,
            evidence_resume=result.get("resume_evidence", []),
            evidence_job=result.get("job_evidence", []),
            final_advice=final_advice or "추가 분석 결과를 생성하지 못했습니다."
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")