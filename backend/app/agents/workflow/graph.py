from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.agents.retrieval_agent import retrieval_agent
from app.agents.generation_agent import generation_agent
from app.agents.supervisor import supervisor_agent


class ChatState(TypedDict):
    question: str
    documents: list
    answer: str
    next: str


def supervisor_node(state: ChatState):
    return supervisor_agent(state)


def retrieve_node(state: ChatState):
    documents = retrieval_agent(state["question"])

    return {
        "documents": documents
    }


def generate_node(state: ChatState):
    answer = generation_agent(
        state["question"],
        state["documents"]
    )

    return {
        "answer": answer
    }


graph_builder = StateGraph(ChatState)


# Add nodes
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)


# START → Supervisor
graph_builder.add_edge(
    START,
    "supervisor"
)


# Supervisor decides next agent
graph_builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "retrieve": "retrieve",
        "generate": "generate",
        "end": END
    }
)


# Retrieval → Supervisor
graph_builder.add_edge(
    "retrieve",
    "supervisor"
)


# Generation → Supervisor
graph_builder.add_edge(
    "generate",
    "supervisor"
)


chat_graph = graph_builder.compile()