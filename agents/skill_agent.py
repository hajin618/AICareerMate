from llm import get_llm
from langchain.prompts import PromptTemplate


class SkillAgent:

    def __init__(self):
        self.llm = get_llm()

        self.prompt = PromptTemplate(
            input_variables=["resume_text", "job_analysis"],
            template="""
You are a career skill analyst.

Compare the resume and job requirements.

Resume:
{resume_text}

Job Analysis:
{job_analysis}

Analyze:

1. Matching skills
2. Missing skills
3. Skill gap priority

Return in Korean.
"""
        )

    def run(self, state: dict):

        resume_text = state["resume_text"]
        job_analysis = state["job_analysis"]

        chain = self.prompt | self.llm

        result = chain.invoke({
            "resume_text": resume_text,
            "job_analysis": job_analysis
        })

        state["skill_analysis"] = result.content

        return state
