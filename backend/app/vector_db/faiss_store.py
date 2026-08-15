import faiss
import numpy as np
import pickle
import os


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../.."
    )
)


# ---------------------------------------------------------
# FAISS storage path
# ---------------------------------------------------------

FAISS_PATH = os.path.join(
    BASE_DIR,
    "backend",
    "data",
    "faiss_index"
)


class FAISSStore:

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []


    # -----------------------------------------------------
    # Add embeddings
    # -----------------------------------------------------

    def add_embeddings(self, embeddings, documents):

        embeddings = np.array(
            embeddings
        ).astype("float32")

        self.index.add(
            embeddings
        )

        self.documents.extend(
            documents
        )


    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    def search(
        self,
        query_embedding,
        top_k=3
    ):

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for idx in indices[0]:

            if idx < len(self.documents):

                results.append(
                    self.documents[idx]
                )

        return results


    # -----------------------------------------------------
    # Save FAISS index
    # -----------------------------------------------------

    def save(
        self,
        path=FAISS_PATH
    ):

        os.makedirs(
            path,
            exist_ok=True
        )

        # Save FAISS index
        faiss.write_index(
            self.index,
            os.path.join(
                path,
                "index.faiss"
            )
        )

        # Save documents
        with open(
            os.path.join(
                path,
                "documents.pkl"
            ),
            "wb"
        ) as f:

            pickle.dump(
                self.documents,
                f
            )


    # -----------------------------------------------------
    # Load FAISS index
    # -----------------------------------------------------

    @classmethod
    def load(
        cls,
        path=FAISS_PATH
    ):

        index_path = os.path.join(
            path,
            "index.faiss"
        )

        documents_path = os.path.join(
            path,
            "documents.pkl"
        )

        # Check files exist
        if not os.path.exists(index_path):

            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not os.path.exists(documents_path):

            raise FileNotFoundError(
                f"Documents file not found: {documents_path}"
            )


        # Load FAISS index
        index = faiss.read_index(
            index_path
        )


        # Load documents
        with open(
            documents_path,
            "rb"
        ) as f:

            documents = pickle.load(
                f
            )


        # Create store
        store = cls(
            index.d
        )

        store.index = index

        store.documents = documents

        return store


# ---------------------------------------------------------
# Create FAISS index
# ---------------------------------------------------------

def create_faiss_index(
    embeddings,
    documents
):

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]


    # Create store
    store = FAISSStore(
        dimension
    )


    # Add embeddings
    store.add_embeddings(
        embeddings,
        documents
    )


    # Save index
    store.save(
        FAISS_PATH
    )


    return len(documents)