import faiss
import numpy as np
import pickle
import os


class FAISSStore:

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add_embeddings(self, embeddings, documents):

        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(self, query_embedding, top_k=3):

        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])

        return results

    def save(self, path="data/faiss_index"):

        os.makedirs(path, exist_ok=True)

        faiss.write_index(
            self.index,
            os.path.join(path, "index.faiss")
        )

        with open(
            os.path.join(path, "documents.pkl"),
            "wb"
        ) as f:
            pickle.dump(
                self.documents,
                f
            )

    @classmethod
    def load(cls, path="data/faiss_index"):

        index = faiss.read_index(
            os.path.join(path, "index.faiss")
        )

        with open(
            os.path.join(path, "documents.pkl"),
            "rb"
        ) as f:
            documents = pickle.load(f)

        store = cls(index.d)

        store.index = index
        store.documents = documents

        return store


def create_faiss_index(embeddings, documents):

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    store = FAISSStore(dimension)

    store.add_embeddings(
        embeddings,
        documents
    )

    # Save FAISS index and document chunks
    store.save("data/faiss_index")

    return len(documents)