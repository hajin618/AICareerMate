from llm import get_llm
from langchain.prompts import PromptTemplate


class JobAgent:

    def __init__(self):
        self.llm = get_llm()

        self.prompt = PromptTemplate(
            input_variables=["job_text"],
            template="""
You are an expert recruiter.

Analyze the following job description and extract:

1. Required skills
2. Preferred skills
3. Experience level
4. Main responsibilities

Job Description:
{job_text}

Return in Korean with bullet points.
"""
        )

    def run(self, state: dict):

        job_text = state["job_text"]

        chain = self.prompt | self.llm

        result = chain.invoke({
            "job_text": job_text
        })

        state["job_analysis"] = result.content

        return state
