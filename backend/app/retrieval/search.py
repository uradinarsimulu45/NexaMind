from app.vector_db.faiss_store import FAISSStore
from app.vector_db.embeddings import generate_embeddings


def search_documents(query, top_k=3):
    # Load saved FAISS index
    store = FAISSStore.load("data/faiss_index")

    # Generate embedding for the question
    query_embedding = generate_embeddings([query])

    # Search
    results = store.search(
        query_embedding,
        top_k
    )

    return results