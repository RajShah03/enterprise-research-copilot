import pickle
from pathlib import Path

import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import embed_model


# ============================================================
# Configuration
# ============================================================

DB_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "financial_docs"

ARTIFACT_DIR = Path("./storage")
BM25_PATH = ARTIFACT_DIR / "bm25.pkl"
NODES_PATH = ARTIFACT_DIR / "nodes.pkl"


# ============================================================
# Load RAG Resources
# ============================================================

def load_rag_resources():
    """
    Loads already-built RAG resources.

    No PDF ingestion or embedding is performed here.

    Returns:
        index
        bm25
        leaf_nodes
    """

    # ========================================================
    # Validate persisted artifacts
    # ========================================================

    if not BM25_PATH.exists():

        raise FileNotFoundError(
            f"BM25 artifact not found: {BM25_PATH}. "
            "Run 'python ingest.py' first."
        )

    if not NODES_PATH.exists():

        raise FileNotFoundError(
            f"Node artifact not found: {NODES_PATH}. "
            "Run 'python ingest.py' first."
        )

    # ========================================================
    # Load Chroma
    # ========================================================

    print(
        "🗄️ Loading persisted Chroma collection..."
    )

    db = chromadb.PersistentClient(
        path=DB_DIR
    )

    try:

        chroma_collection = db.get_collection(
            CHROMA_COLLECTION_NAME
        )

    except Exception as exc:

        raise FileNotFoundError(
            f"Chroma collection "
            f"'{CHROMA_COLLECTION_NAME}' "
            f"was not found. "
            f"Run 'python ingest.py' first."
        ) from exc

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    # ========================================================
    # Reconnect LlamaIndex to existing vectors
    # ========================================================

    print(
        "🧠 Connecting LlamaIndex to persisted vectors..."
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # ========================================================
    # Load BM25
    # ========================================================

    print(
        "🔍 Loading persisted BM25 index..."
    )

    with BM25_PATH.open(
        "rb"
    ) as file:

        bm25 = pickle.load(
            file
        )

    # ========================================================
    # Load nodes
    # ========================================================

    print(
        "🧩 Loading persisted node hierarchy..."
    )

    with NODES_PATH.open(
        "rb"
    ) as file:

        node_artifacts = pickle.load(
            file
        )

    leaf_nodes = node_artifacts[
        "leaf_nodes"
    ]

    print(
        f"✅ Loaded "
        f"{len(leaf_nodes)} "
        f"leaf nodes."
    )

    print(
        "✅ Persisted RAG resources loaded successfully."
    )

    return (
        index,
        bm25,
        leaf_nodes,
    )