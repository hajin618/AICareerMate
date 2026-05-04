from langgraph.graph import StateGraph

class AgentState(dict):
    pass


def build_graph(resume_agent, job_agent, skill_agent, coach_agent):

    graph = StateGraph(AgentState)

    graph.add_node("resume", resume_agent.run)
    graph.add_node("job", job_agent.run)
    graph.add_node("skill", skill_agent.run)
    graph.add_node("coach", coach_agent.run)

    graph.set_entry_point("resume")

    graph.add_edge("resume", "job")
    graph.add_edge("job", "skill")
    graph.add_edge("skill", "coach")

    graph.set_finish_point("coach")

    return graph.compile()

def check_resume_quality(state):
    resume_text = state.get("resume_text", "")

    if len(resume_text.strip()) < 100:
        return "retry"
    return "next"