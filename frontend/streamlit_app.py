import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

API_URL = "https://nexamind-backend-8vbm.onrender.com"


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
        timeout=60
    )

    if response.status_code == 200:

        st.success(
            "🟢 OmniBrain backend is online"
        )

    else:

        st.warning(
            f"Backend returned HTTP {response.status_code}"
        )

except requests.exceptions.RequestException as e:

    st.error(
        "🔴 Could not connect to the OmniBrain backend."
    )

    st.caption(
        f"Backend: {API_URL}"
    )

    st.exception(e)

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "document_id" not in st.session_state:
    st.session_state["document_id"] = None

if "document_processed" not in st.session_state:
    st.session_state["document_processed"] = False

if "upload_result" not in st.session_state:
    st.session_state["upload_result"] = None


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
                    timeout=600
                )

                # --------------------------------------------
                # SUCCESS
                # --------------------------------------------

                if response.status_code == 200:

                    result = response.json()

                    document_id = result.get(
                        "document_id"
                    )

                    if not document_id:

                        st.error(
                            "Upload succeeded, but no "
                            "document_id was returned."
                        )

                    else:

                        # Save document information
                        st.session_state[
                            "document_id"
                        ] = document_id

                        st.session_state[
                            "document_processed"
                        ] = True

                        st.session_state[
                            "upload_result"
                        ] = result

                        st.success(
                            "✅ PDF processed successfully!"
                        )

                        st.info(
                            f"Document ID: {document_id}"
                        )

                else:

                    st.error(
                        f"Upload failed: "
                        f"HTTP {response.status_code}"
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
# DOCUMENT INFORMATION
# ============================================================

if st.session_state["upload_result"] is not None:

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
            result.get(
                "pages",
                0
            )
        )

    with col2:

        st.metric(
            "Images",
            result.get(
                "images",
                0
            )
        )

    with col3:

        st.metric(
            "Chunks",
            result.get(
                "chunks",
                0
            )
        )

    with col4:

        st.metric(
            "Vectors",
            result.get(
                "stored_vectors",
                0
            )
        )

    st.caption(
        f"Current document: "
        f"{result.get('filename', 'Unknown')}"
    )


# ============================================================
# QUESTION
# ============================================================

st.divider()

st.header(
    "💬 Ask a question"
)

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

    elif not st.session_state.get(
        "document_id"
    ):

        st.warning(
            "Please upload and process a PDF first."
        )

    else:

        document_id = st.session_state[
            "document_id"
        ]

        with st.spinner(
            "🧠 OmniBrain is thinking..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": question,
                        "document_id": document_id
                    },
                    timeout=600
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
                            "Document ID:",
                            result.get(
                                "document_id",
                                document_id
                            )
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
                        f"HTTP {response.status_code}"
                    )

                    st.code(
                        response.text
                    )

            except requests.exceptions.RequestException as e:

                st.error(
                    "❌ Could not connect to the OmniBrain backend."
                )

                st.exception(e)