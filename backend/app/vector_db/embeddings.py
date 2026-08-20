import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"

model = None


def load_model():
    global model

    if model is None:

        print("Loading embedding model...")

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            MODEL_NAME
        )

        print("Embedding model loaded.")


def generate_embeddings(text_chunks):
    """
    Generate embeddings for text chunks.
    """

    load_model()

    embeddings = model.encode(
        text_chunks,
        convert_to_numpy=True
    )

    return embeddings


def generate_query_embedding(query):
    """
    Generate embedding for user query.
    """

    load_model()

    embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    return embedding