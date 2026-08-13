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
# SUPERVISOR ROUTER
# -----------------------------
def supervisor_router(state: ChatState):
    """
    Decide which node should execute next.

    Routing:
        No documents -> retrieve
        Documents but no answer -> generate
        Answer exists -> end
    """

    documents = state.get("documents", [])
    answer = state.get("answer", "")

    # Step 1: Retrieve relevant documents
    if not documents:
        return "retrieve"

    # Step 2: Generate answer from retrieved documents
    if not answer:
        return "generate"

    # Step 3: Finish workflow
    return "end"


# -----------------------------
# Build graph
# -----------------------------
graph_builder = StateGraph(ChatState)


# -----------------------------
# Add nodes
# -----------------------------
graph_builder.add_node(
    "retrieve",
    retrieve_node
)

graph_builder.add_node(
    "generate",
    generate_node
)


# -----------------------------
# START -> Supervisor
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
# Retrieve -> Supervisor
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
# Generate -> END
# -----------------------------
graph_builder.add_edge(
    "generate",
    END
)


# -----------------------------
# Compile
# -----------------------------
chat_graph = graph_builder.compile()