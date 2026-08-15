from pathlib import Path

from app.vector_db.faiss_store import FAISSStore
from app.vector_db.embeddings import generate_embeddings


# backend/
BASE_DIR = Path(__file__).resolve().parents[2]

FAISS_PATH = BASE_DIR / "data" / "faiss_index"


def search_documents(query, top_k=5):
    """
    Search FAISS for the most relevant PDF pages.
    """

    # Load FAISS index
    store = FAISSStore.load(str(FAISS_PATH))

    # Create embedding for the question
    query_embedding = generate_embeddings([query])

    # Search FAISS
    results = store.search(
        query_embedding,
        top_k
    )

    return results