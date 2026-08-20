import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000"


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

st.subheader(
    "Agentic Multi-Modal RAG Orchestrator"
)

st.write(
    "Upload a document and ask questions about its contents."
)

st.divider()


# ============================================================
# BACKEND HEALTH CHECK
# ============================================================

try:

    response = requests.get(
        f"{API_URL}/",
        timeout=5
    )

    if response.status_code == 200:
        st.success("🟢 OmniBrain backend is online")

    else:
        st.warning("Backend responded with an unexpected status.")

except requests.exceptions.RequestException:

    st.error(
        "🔴 Backend is not running. "
        "Start FastAPI on port 8000."
    )

    st.stop()


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("📄 Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    st.write(
        f"Selected file: **{uploaded_file.name}**"
    )

    if st.button(
        "📥 Process PDF",
        use_container_width=True
    ):

        with st.spinner(
            "Processing PDF..."
        ):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                }

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    timeout=300
                )


                # --------------------------------------------
                # SUCCESS
                # --------------------------------------------

                if response.status_code == 200:

                    result = response.json()

                    st.session_state[
                        "document_processed"
                    ] = True

                    st.session_state[
                        "upload_result"
                    ] = result

                    st.success(
                        "✅ PDF processed successfully!"
                    )

                else:

                    st.error(
                        f"Upload failed: "
                        f"{response.status_code}"
                    )

                    st.code(
                        response.text
                    )

            except requests.exceptions.RequestException as e:

                st.error(
                    "❌ Could not connect to FastAPI."
                )

                st.exception(e)


# ============================================================
# SHOW PROCESSING INFORMATION
# ============================================================

if "upload_result" in st.session_state:

    result = st.session_state[
        "upload_result"
    ]

    st.divider()

    st.subheader(
        "📊 Document Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Pages",
            result.get("pages", 0)
        )

    with col2:
        st.metric(
            "Images",
            result.get("images", 0)
        )

    with col3:
        st.metric(
            "Chunks",
            result.get("chunks", 0)
        )

    with col4:
        st.metric(
            "Vectors",
            result.get("stored_vectors", 0)
        )


# ============================================================
# QUESTION
# ============================================================

st.divider()

st.header("💬 Ask a question")

question = st.text_input(
    "Question",
    placeholder=(
        "Ask something about your uploaded document..."
    )
)


# ============================================================
# ASK OMNIBRAIN
# ============================================================

if st.button(
    "🚀 Ask OmniBrain",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif "document_processed" not in st.session_state:

        st.warning(
            "Please upload and process a PDF first."
        )

    else:

        with st.spinner(
            "🧠 OmniBrain is thinking..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": question
                    },
                    timeout=300
                )


                # ----------------------------------------
                # SUCCESS
                # ----------------------------------------

                if response.status_code == 200:

                    result = response.json()

                    answer = result.get(
                        "answer",
                        ""
                    )

                    st.divider()

                    st.subheader(
                        "🤖 Answer"
                    )

                    if answer:

                        st.success(
                            answer
                        )

                    else:

                        st.warning(
                            "No answer was generated."
                        )


                    # ----------------------------------------
                    # DEBUG
                    # ----------------------------------------

                    with st.expander(
                        "🔧 Debug information"
                    ):

                        st.write(
                            "Question:",
                            question
                        )

                        st.write(
                            "Retrieved chunks:",
                            result.get(
                                "retrieved_chunks",
                                0
                            )
                        )

                        st.write(
                            "Conversation length:",
                            result.get(
                                "conversation_length",
                                0
                            )
                        )

                else:

                    st.error(
                        f"Chat request failed: "
                        f"{response.status_code}"
                    )

                    st.code(
                        response.text
                    )

            except requests.exceptions.RequestException as e:

                st.error(
                    "❌ Could not connect to FastAPI."
                )

                st.exception(e)