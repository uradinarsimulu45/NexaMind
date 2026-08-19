import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="OmniBrain",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🧠 OmniBrain")

st.subheader("Agentic Multi-Modal RAG Orchestrator")

st.write(
    "Ask questions about your documents and let OmniBrain "
    "retrieve the relevant information and generate an answer."
)

st.divider()


# ============================================================
# LOAD LANGGRAPH ONLY WHEN NEEDED
# ============================================================

@st.cache_resource(show_spinner=False)
def load_chat_graph():
    """
    Load LangGraph only once.

    This prevents the Hugging Face / transformer models
    from loading every time Streamlit reruns the page.
    """

    from app.agents.workflow.graph import chat_graph

    return chat_graph


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload a PDF",
    type=["pdf"],
)

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # Save uploaded PDF temporarily
    upload_dir = ROOT_DIR / "data" / "uploads"
    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = upload_dir / uploaded_file.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(
        f"PDF saved successfully: {pdf_path.name}"
    )


st.divider()


# ============================================================
# QUESTION
# ============================================================

question = st.text_input(
    "💬 Ask a question",
    placeholder=(
        "Example: How much funding is requested "
        "for Gateway development?"
    ),
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🚀 Ask OmniBrain",
    type="primary",
    use_container_width=True,
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # ----------------------------------------------------
        # LOAD GRAPH
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Loading OmniBrain..."
        ):

            try:

                chat_graph = load_chat_graph()

            except Exception as e:

                st.error(
                    "Failed to load OmniBrain."
                )

                st.exception(e)

                st.stop()


        # ----------------------------------------------------
        # RUN QUERY
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Searching documents and generating answer..."
        ):

            try:

                result = chat_graph.invoke(
                    {
                        "question": question,
                        "documents": [],
                        "answer": "",
                        "history": [],
                        "vision_result": "",
                    }
                )

                # --------------------------------------------
                # GET ANSWER
                # --------------------------------------------

                answer = result.get(
                    "answer",
                    "",
                )

                st.divider()

                st.subheader(
                    "🤖 Answer"
                )

                if answer:

                    st.success(answer)

                else:

                    st.warning(
                        "The answer is not available "
                        "in the provided documents."
                    )


                # --------------------------------------------
                # DEBUG INFORMATION
                # --------------------------------------------

                with st.expander(
                    "🔧 Debug information"
                ):

                    st.write(
                        "Question:",
                        question,
                    )

                    st.write(
                        "Result:",
                        result,
                    )


            except Exception as e:

                st.error(
                    "❌ OmniBrain error"
                )

                st.exception(e)