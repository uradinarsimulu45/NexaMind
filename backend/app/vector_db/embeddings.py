from sentence_transformers import SentenceTransformer
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(text_chunks):
    """
    Generate embeddings for text chunks.

    Args:
        text_chunks: list of text chunks

    Returns:
        numpy array of embeddings
    """

    embeddings = model.encode(
        text_chunks,
        convert_to_numpy=True
    )

    return embeddings


def generate_query_embedding(query):
    """
    Generate embedding for user query.
    """

    embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    return embedding