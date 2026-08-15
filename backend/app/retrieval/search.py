from pathlib import Path

from app.vector_db.faiss_store import FAISSStore
from app.vector_db.embeddings import generate_embeddings


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# FAISS index is currently inside backend/data
FAISS_PATH = PROJECT_ROOT / "backend" / "data" / "faiss_index"


def search_documents(query, top_k=5):
    """
    Search FAISS for the most relevant PDF pages.
    """

    store = FAISSStore.load(str(FAISS_PATH))

    query_embedding = generate_embeddings([query])

    results = store.search(
        query_embedding,
        top_k
    )

    return results