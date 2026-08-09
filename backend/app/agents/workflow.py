from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.retrieval.search import search_documents
from app.retrieval.prompt_builder import build_prompt
from app.llm.generator import generate_answer
from app.memory.conversation import ConversationMemory


class AgentState(TypedDict):
    question: str
    documents: list
    prompt: str
    answer: str


memory = ConversationMemory()


def retrieve_documents(state: AgentState):
    """Retrieve relevant documents using the existing semantic search."""

    documents = search_documents(state["question"])

    return {
        "documents": documents
    }


def create_prompt(state: AgentState):
    """Build the LLM prompt using the retrieved documents."""

    prompt = build_prompt(
        state["question"],
        state["documents"]
    )

    return {
        "prompt": prompt
    }


def generate_response(state: AgentState):
    """Generate the final answer using the existing local LLM."""

    answer = generate_answer(state["prompt"])

    return {
        "answer": answer
    }


# Create LangGraph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("retrieve", retrieve_documents)
graph.add_node("build_prompt", create_prompt)
graph.add_node("generate", generate_response)

# Define workflow
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "build_prompt")
graph.add_edge("build_prompt", "generate")
graph.add_edge("generate", END)

# Compile
workflow = graph.compile()