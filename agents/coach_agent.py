from llm import get_llm
from langchain.prompts import PromptTemplate


class CoachAgent:

    def __init__(self):
        self.llm = get_llm()

        self.prompt = PromptTemplate(
            input_variables=["job_analysis", "skill_analysis"],
            template="""
You are a professional career coach.

Based on the analysis below, give personalized advice.

Job Analysis:
{job_analysis}

Skill Gap Analysis:
{skill_analysis}

Provide:

1. Resume improvement tips
2. Study roadmap (3 months)
3. Interview preparation advice
4. Career strategy

Write in Korean, structured format.
"""
        )

    def run(self, state: dict):

        job_analysis = state["job_analysis"]
        skill_analysis = state["skill_analysis"]

        chain = self.prompt | self.llm

        result = chain.invoke({
            "job_analysis": job_analysis,
            "skill_analysis": skill_analysis
        })

        state["final_advice"] = result.content

        return state
