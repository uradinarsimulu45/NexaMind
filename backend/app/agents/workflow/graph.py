from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.agents.retrieval_agent import retrieval_agent
from app.agents.generation_agent import generation_agent


class ChatState(TypedDict):
    question: str
    documents: list
    answer: str
    history: list


# -----------------------------
# Retrieval node
# -----------------------------
def retrieve_node(state: ChatState):
    documents = retrieval_agent(
        state["question"]
    )

    return {
        "documents": documents
    }


# -----------------------------
# Generation node
# -----------------------------
def generate_node(state: ChatState):
    answer = generation_agent(
        state["question"],
        state["documents"]
    )

    return {
        "answer": answer
    }


# -----------------------------
# Supervisor ROUTER
# -----------------------------
def supervisor_router(state: ChatState):

    # No documents yet → retrieve
    if not state.get("documents"):
        return "retrieve"

    # Documents exist but no answer → generate
    if not state.get("answer"):
        return "generate"

    # Answer exists → finish
    return "end"


# -----------------------------
# Build graph
# -----------------------------
graph_builder = StateGraph(ChatState)


# Add only REAL state-update nodes
graph_builder.add_node(
    "retrieve",
    retrieve_node
)

graph_builder.add_node(
    "generate",
    generate_node
)


# -----------------------------
# START → Supervisor router
# -----------------------------
graph_builder.add_conditional_edges(
    START,
    supervisor_router,
    {
        "retrieve": "retrieve",
        "generate": "generate",
        "end": END
    }
)


# -----------------------------
# Retrieval → Supervisor router
# -----------------------------
graph_builder.add_conditional_edges(
    "retrieve",
    supervisor_router,
    {
        "retrieve": "retrieve",
        "generate": "generate",
        "end": END
    }
)


# -----------------------------
# Generation → END
# -----------------------------
graph_builder.add_edge(
    "generate",
    END
)


# Compile
chat_graph = graph_builder.compile()