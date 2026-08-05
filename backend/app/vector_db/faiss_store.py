import os
import faiss
import numpy as np

VECTOR_PATH = "data/vectors"

os.makedirs(VECTOR_PATH, exist_ok=True)


def create_faiss_index(embeddings):
    """
    Create a FAISS index and save it.
    """

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    faiss.write_index(
        index,
        os.path.join(VECTOR_PATH, "faiss_index.bin")
    )

    return index.ntotal