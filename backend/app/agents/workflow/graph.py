from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.agents.retrieval_agent import retrieval_agent
from app.agents.generation_agent import generation_agent
from app.agents.vision.vision_agent import vision_agent


# ---------------------------------
# Project root
# ---------------------------------

ROOT_DIR = Path(__file__).resolve().parents[4]


# ---------------------------------
# Chat state
# ---------------------------------

class ChatState(TypedDict):
    question: str
    documents: list
    answer: str
    history: list
    vision_result: str


# ---------------------------------
# Retrieval node
# ---------------------------------

def retrieve_node(state: ChatState):

    documents = retrieval_agent(
        state["question"]
    )

    return {
        "documents": documents
    }


# ---------------------------------
# Vision node
# ---------------------------------

def vision_node(state: ChatState):

    image_path = (
        ROOT_DIR
        / "backend"
        / "data"
        / "images"
        / "page_14_img_65.jpeg"
    )

    # Check image exists
    if not image_path.exists():
        return {
            "vision_result": (
                f"Vision image not found: {image_path}"
            )
        }

    result = vision_agent(
        str(image_path)
    )

    return {
        "vision_result": result
    }


# ---------------------------------
# Generation node
# ---------------------------------

def generate_node(state: ChatState):

    documents = state.get(
        "documents",
        []
    )

    vision_result = state.get(
        "vision_result",
        ""
    )

    # ---------------------------------
    # Add visual evidence
    # ---------------------------------

    if vision_result:

        documents = documents + [
            {
                "text": (
                    f"Visual information: "
                    f"{vision_result}"
                ),
                "source": "vision_agent",
                "page": 14
            }
        ]

    # ---------------------------------
    # Generate answer
    # ---------------------------------

    answer = generation_agent(
        state["question"],
        documents
    )

    return {
        "answer": answer
    }


# ---------------------------------
# Supervisor router
# ---------------------------------

def supervisor_router(
    state: ChatState
):

    question = state["question"].lower()

    # ---------------------------------
    # Visual question detection
    # ---------------------------------

    visual_keywords = [
        "image",
        "picture",
        "photo",
        "figure",
        "diagram",
        "chart",
        "graph",
        "visual",
        "shown",
        "spacecraft",
        "planet",
        "rocket",
        "satellite",
        "moon"
    ]

    is_visual_question = any(
        word in question
        for word in visual_keywords
    )

    # ---------------------------------
    # Visual workflow
    # ---------------------------------

    if is_visual_question:

        if not state.get(
            "vision_result"
        ):
            return "vision"

        if not state.get(
            "answer"
        ):
            return "generate"

        return "end"

    # ---------------------------------
    # Normal RAG workflow
    # ---------------------------------

    if not state.get(
        "documents"
    ):
        return "retrieve"

    if not state.get(
        "answer"
    ):
        return "generate"

    return "end"


# ---------------------------------
# Build LangGraph
# ---------------------------------

graph_builder = StateGraph(
    ChatState
)


# ---------------------------------
# Add nodes
# ---------------------------------

graph_builder.add_node(
    "retrieve",
    retrieve_node
)

graph_builder.add_node(
    "vision",
    vision_node
)

graph_builder.add_node(
    "generate",
    generate_node
)


# ---------------------------------
# START → Supervisor
# ---------------------------------

graph_builder.add_conditional_edges(
    START,
    supervisor_router,
    {
        "retrieve": "retrieve",
        "vision": "vision",
        "generate": "generate",
        "end": END
    }
)


# ---------------------------------
# Retrieval → Supervisor
# ---------------------------------

graph_builder.add_conditional_edges(
    "retrieve",
    supervisor_router,
    {
        "retrieve": "retrieve",
        "vision": "vision",
        "generate": "generate",
        "end": END
    }
)


# ---------------------------------
# Vision → Supervisor
# ---------------------------------

graph_builder.add_conditional_edges(
    "vision",
    supervisor_router,
    {
        "retrieve": "retrieve",
        "vision": "vision",
        "generate": "generate",
        "end": END
    }
)


# ---------------------------------
# Generation → END
# ---------------------------------

graph_builder.add_edge(
    "generate",
    END
)


# ---------------------------------
# Compile graph
# ---------------------------------

chat_graph = graph_builder.compile()