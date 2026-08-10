from app.retrieval.search import search_documents


def retrieval_agent(question: str):
    """
    Retrieval Agent

    Searches the FAISS vector database for
    relevant documents.
    """

    documents = search_documents(question)

    return documents