# Day 6: Added SentenceTransformer embeddings using all-MiniLM-L6-v2
from sentence_transformers import SentenceTransformer

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):
    """
    Generate embeddings for text chunks.
    """
    embeddings = model.encode(chunks)

    return embeddings