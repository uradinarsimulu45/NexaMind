🧠 Nexamind

Agentic Multi-Modal RAG Orchestrator

Nexamind is an Agentic Multi-Modal Retrieval-Augmented Generation (RAG) system designed to answer questions from PDF documents. It combines document ingestion, text extraction, semantic embeddings, FAISS vector search, LangGraph-based orchestration, and an LLM generation layer through a FastAPI backend and Streamlit frontend.

The project is designed as a college-level AI/ML project demonstrating how an agentic RAG pipeline can combine multiple components into one document-question-answering application.

✨ Features

📄 Upload PDF documents through a Streamlit interface

🔎 Extract PDF text page-by-page using PyMuPDF

🧩 Treat extracted pages as searchable document chunks

🧠 Generate semantic embeddings using all-MiniLM-L6-v2

⚡ Store and search embeddings using FAISS

🤖 Generate grounded answers using FLAN-T5

🕸️ Orchestrate the workflow using LangGraph

🔍 Retrieval Agent for relevant document chunks

✍️ Generation Agent for grounded final answers

👁️ Vision Agent foundation for image understanding using BLIP

💬 Conversation memory for multi-turn interaction

🚀 FastAPI REST endpoints for upload and chat

🖥️ Streamlit frontend for user interaction

🧪 Local end-to-end testing with real PDF questions

🏗️ System Architecture

                    ┌───────────────────────────┐
                    │       Streamlit UI         │
                    │   Upload PDF / Ask Query   │
                    └─────────────┬─────────────┘
                                  │ HTTP
                                  ▼
                    ┌───────────────────────────┐
                    │       FastAPI Backend      │
                    │       /upload /chat        │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │       PDF Processing       │
                    │ Text extraction + chunks  │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │      Embedding Model       │
                    │   all-MiniLM-L6-v2        │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │          FAISS             │
                    │    Vector Search Index     │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │        LangGraph           │
                    │      Orchestration         │
                    └─────────────┬─────────────┘
                                  │
                       ┌──────────┴──────────┐
                       ▼                     ▼
              ┌────────────────┐    ┌────────────────┐
              │ Retrieval Agent│    │  Vision Agent  │
              └───────┬────────┘    └───────┬────────┘
                      │                     │
                      └──────────┬──────────┘
                                 ▼
                       ┌──────────────────┐
                       │ Generation Agent │
                       │     FLAN-T5      │
                       └────────┬─────────┘
                                ▼
                       ┌──────────────────┐
                       │   Final Answer   │
                       └──────────────────┘

🔄 End-to-End Workflow

1. Document Upload

The user uploads a PDF from the Streamlit frontend.

2. PDF Processing

The FastAPI /upload endpoint:

Saves the PDF.

Extracts text page-by-page.

Creates page-level document chunks.

Generates embeddings.

Stores the vectors in FAISS.

Returns processing information.

3. Question

The user submits a natural-language question through the Streamlit interface.

4. Retrieval

The Retrieval Agent converts the question into an embedding and searches the FAISS index for the most relevant document chunks.

5. Generation

The Generation Agent builds a grounded prompt using the retrieved evidence and generates the final answer with FLAN-T5.

6. Conversation Memory

The conversation memory component keeps previous questions and answers available to the application for multi-turn interaction.

📁 Project Structure

NexaMind/
│
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── generation_agent.py
│   │   │   ├── retrieval_agent.py
│   │   │   ├── supervisor.py
│   │   │   └── vision/
│   │   │       ├── image_selector.py
│   │   │       └── vision_agent.py
│   │   │
│   │   ├── api/
│   │   │   ├── chat.py
│   │   │   └── upload.py
│   │   │
│   │   ├── llm/
│   │   │   └── generator.py
│   │   │
│   │   ├── memory/
│   │   │   └── conversation.py
│   │   │
│   │   ├── retrieval/
│   │   │   ├── prompt_builder.py
│   │   │   └── search.py
│   │   │
│   │   ├── utils/
│   │   │   ├── extract_images.py
│   │   │   ├── extract_text.py
│   │   │   └── text_chunker.py
│   │   │
│   │   ├── vector_db/
│   │   │   ├── embeddings.py
│   │   │   └── faiss_store.py
│   │   │
│   │   └── main.py
│   │
│   ├── data/
│   │   ├── faiss_index/
│   │   ├── images/
│   │   └── pdfs/
│   │
│   └── requirements.txt
│
├── frontend/
│   └── streamlit_app.py
│
└── README.md

🧰 Technologies Used

Technology

Purpose

Python

Core development language

FastAPI

Backend REST API

Streamlit

Frontend UI

LangGraph

Agent/workflow orchestration

FAISS

Vector similarity search

SentenceTransformers

Text embeddings

FLAN-T5

Local answer generation

PyMuPDF

PDF text/image extraction

Pillow

Image processing

Pydantic

API request validation

Git/GitHub

Version control

⚙️ Installation

1. Clone the repository

git clone https://github.com/uradinarsimulu45/NexaMind.git
cd NexaMind

2. Create a virtual environment

Windows PowerShell:

python -m venv venv
.\venv\Scripts\Activate.ps1

3. Install dependencies

python -m pip install -r backend\requirements.txt

▶️ Run the Backend

Open a terminal and run:

cd NexaMind
.\venv\Scripts\Activate.ps1
$env:PYTHONPATH="backend"
python -m uvicorn app.main:app --reload --port 8000

The API will be available at:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

▶️ Run the Frontend

Open a second terminal:

cd NexaMind
.\venv\Scripts\Activate.ps1
python -m streamlit run frontend\streamlit_app.py

Open:

http://localhost:8501

🔌 API Endpoints

GET /

Health/welcome endpoint.

Example response:

{
  "message": "Welcome to OmniBrain API"
}

POST /upload

Uploads and processes a PDF document.

Example response:

{
  "message": "PDF uploaded successfully",
  "filename": "example.pdf",
  "pages": 42,
  "images": 0,
  "chunks": 42,
  "stored_vectors": 42
}

POST /chat

Accepts a natural-language question and returns a grounded answer based on retrieved document evidence.

Example request:

{
  "question": "How much funding is requested for Gateway development?"
}

Example response:

{
  "question": "How much funding is requested for Gateway development?",
  "retrieved_chunks": 5,
  "answer": "$818M",
  "conversation_length": 1
}

🧪 Example Demonstration

The system was tested locally using NASA's FY 2025 Budget Request PDF.

Example 1:

Question:
How much funding is requested for Gateway development?

Answer:
$818M

Example 2:

Question:
How much funding is requested for the Human Landing System?

Answer:
$1,896M

These tests verified that the application could retrieve relevant document chunks and generate grounded answers.

🔐 Environment and Secrets

Keep secrets out of GitHub.

For local development, use a .env file when secrets are required:

OPENAI_API_KEY=your_key_here

Never commit the real key. Add environment files to .gitignore:

.env
*.env

The current college-project version primarily uses the local FLAN-T5 generation path. Any experimental API-based generator changes should remain outside the stable submission version unless explicitly integrated and tested.

☁️ Deployment Note

A cloud deployment was attempted using Render. The backend successfully started and served health requests, but the selected free instance repeatedly exceeded its 512 MB RAM limit during ML workloads.

The local application was therefore treated as the stable demonstration environment for the college project.

This is a hosting-resource limitation, not a failure of the core RAG implementation.

🚀 Future Enhancements

True multi-document isolation and document management

Dynamic image selection for visual questions

Fully integrated vision + text evidence fusion

More robust chunking strategies

Persistent document metadata/database

Streaming responses

Authentication and user-specific document collections

Production-grade vector database such as Qdrant

Cloud deployment on a host with sufficient RAM/GPU resources

🎓 Academic Project Summary

Project Title: OmniBrain – Agentic Multi-Modal RAG Orchestrator

Domain: Generative AI / Agentic AI / Retrieval-Augmented Generation

Problem Addressed: Traditional RAG pipelines can struggle with large and multimodal PDF documents because useful information may exist in text, tables, figures, and images.

Proposed Solution: OmniBrain combines PDF processing, embeddings, vector retrieval, LangGraph orchestration, retrieval and generation agents, and a vision-agent foundation to provide grounded question answering over documents.

Primary Outcome: A functional end-to-end local prototype demonstrating document ingestion, semantic retrieval, agentic orchestration, and grounded answer generation.

👤 Author

URADI NARSIMULU
B.Tech – Computer Science & Engineering

GitHub: uradinarsimulu45

📄 License

This project is intended primarily for educational and academic demonstration purposes.
