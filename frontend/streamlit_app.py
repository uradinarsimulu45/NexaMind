import sys
from pathlib import Path

import streamlit as st


# ---------------------------------
# Add backend to Python path
# ---------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"

sys.path.insert(0, str(BACKEND_DIR))


# ---------------------------------
# Import LangGraph
# ---------------------------------

from app.agents.workflow.graph import chat_graph


# ---------------------------------
# Streamlit configuration
# ---------------------------------

st.set_page_config(
    page_title="OmniBrain",
    page_icon="🧠",
    layout="wide"
)


# ---------------------------------
# Header
# ---------------------------------

st.title("🧠 OmniBrain")

st.subheader(
    "Agentic Multi-Modal RAG Orchestrator"
)

st.write(
    "Ask questions and let OmniBrain route them "
    "through Retrieval, Vision, and Generation agents."
)

st.divider()


# ---------------------------------
# PDF Upload
# ---------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload a PDF",
    type=["pdf"]
)

if uploaded_file:
    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


st.divider()


# ---------------------------------
# Question
# ---------------------------------

question = st.text_input(
    "💬 Ask a question",
    placeholder=(
        "Example: How much funding is requested "
        "for Gateway development?"
    )
)


# ---------------------------------
# Ask OmniBrain
# ---------------------------------

if st.button("Ask OmniBrain"):

    if not question:
        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "OmniBrain is thinking..."
        ):

            try:

                result = chat_graph.invoke(
                    {
                        "question": question,
                        "documents": [],
                        "answer": "",
                        "history": [],
                        "vision_result": ""
                    }
                )

                answer = result.get(
                    "answer",
                    ""
                )

                st.divider()

                st.subheader("🤖 Answer")

                st.success(answer)

            except Exception as e:

                st.error(
                    f"OmniBrain error: {e}"
                )