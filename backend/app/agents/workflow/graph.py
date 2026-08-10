from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.agents.retrieval_agent import retrieval_agent
from app.agents.generation_agent import generation_agent
from app.agents.supervisor import supervisor_agent


class ChatState(TypedDict):
    question: str
    documents: list
    answer: str


def retrieve_node(state: ChatState):

    documents = retrieval_agent(
        state["question"]
    )

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


def supervisor_node(state: ChatState):

    return supervisor_agent(state)


graph_builder = StateGraph(ChatState)

# Agents
graph_builder.add_node(
    "retrieve",
    retrieve_node
)

graph_builder.add_node(
    "generate",
    generate_node
)

# Start
graph_builder.add_edge(
    START,
    "retrieve"
)

# Retrieval → Generation
graph_builder.add_edge(
    "retrieve",
    "generate"
)

# Generation → End
graph_builder.add_edge(
    "generate",
    END
)

chat_graph = graph_builder.compile()