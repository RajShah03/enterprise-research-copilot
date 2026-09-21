import pickle
import re
from pathlib import Path

import chromadb

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)

from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
)

from llama_index.vector_stores.chroma import (
    ChromaVectorStore,
)

from llama_index.core.storage.docstore import (
    SimpleDocumentStore,
)

from rank_bm25 import BM25Okapi

from app.config import embed_model


# ============================================================
# Configuration
# ============================================================

DB_DIR = "./chroma_db"
DATA_DIR = "./data"

CHROMA_COLLECTION_NAME = "financial_docs"

ARTIFACT_DIR = Path("./storage")
BM25_PATH = ARTIFACT_DIR / "bm25.pkl"
NODES_PATH = ARTIFACT_DIR / "nodes.pkl"


# ============================================================
# BM25 Tokenizer
# ============================================================

def tokenize_for_bm25(text: str):
    """
    Tokenizes financial text while preserving:
    - numbers
    - percentages
    - decimals
    - hyphenated terms
    - alphanumeric terms
    """

    if not text:
        return []

    text = text.lower()

    return re.findall(
        r"""
        \d+(?:[.,]\d+)*%?
        |
        [a-zA-Z]+(?:[-_][a-zA-Z0-9]+)*
        |
        [a-zA-Z]+\d+
        """,
        text,
        flags=re.VERBOSE,
    )


# ============================================================
# Reset Chroma Collection
# ============================================================

def _reset_chroma_collection(db):
    """
    Rebuilds the financial document collection from scratch.

    This prevents old vectors from remaining when the source
    documents are re-ingested.
    """

    existing_collections = {
        collection.name
        for collection in db.list_collections()
    }

    if CHROMA_COLLECTION_NAME in existing_collections:

        print(
            f"🗑️ Removing existing Chroma collection: "
            f"{CHROMA_COLLECTION_NAME}"
        )

        db.delete_collection(
            CHROMA_COLLECTION_NAME
        )

    return db.create_collection(
        CHROMA_COLLECTION_NAME
    )


# ============================================================
# Save BM25 + Node Artifacts
# ============================================================

def _save_artifacts(
    bm25,
    nodes,
    leaf_nodes,
):
    """
    Persists the non-Chroma RAG artifacts needed by the
    serving application.

    Full nodes are saved so that parent expansion can be added
    later without re-ingesting the documents.
    """

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with BM25_PATH.open(
        "wb"
    ) as file:

        pickle.dump(
            bm25,
            file,
        )

    with NODES_PATH.open(
        "wb"
    ) as file:

        pickle.dump(
            {
                "nodes": nodes,
                "leaf_nodes": leaf_nodes,
            },
            file,
        )

    print(
        f"💾 Saved BM25 index: {BM25_PATH}"
    )

    print(
        f"💾 Saved node artifacts: {NODES_PATH}"
    )


# ============================================================
# Ingestion
# ============================================================

def run_ingestion():

    print(
        "\n📥 Starting document ingestion..."
    )

    # ========================================================
    # 1. Load documents
    # ========================================================

    print(
        "📂 Loading documents from data/..."
    )

    reader = SimpleDirectoryReader(
        DATA_DIR
    )

    documents = reader.load_data()

    if not documents:

        raise ValueError(
            "No documents found in data/ directory."
        )

    print(
        f"📄 Loaded {len(documents)} document(s)."
    )

    # ========================================================
    # 2. Hierarchical chunking
    # ========================================================

    print(
        "🧩 Creating hierarchical parent-child nodes..."
    )

    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[
            1024,
            128,
        ]
    )

    nodes = node_parser.get_nodes_from_documents(
        documents
    )

    leaf_nodes = get_leaf_nodes(
        nodes
    )

    print(
        f"🧩 Total hierarchical nodes: "
        f"{len(nodes)}"
    )

    print(
        f"🔹 Leaf nodes: "
        f"{len(leaf_nodes)}"
    )

    # ========================================================
    # 3. ChromaDB
    # ========================================================

    print(
        "🗄️ Connecting to ChromaDB..."
    )

    db = chromadb.PersistentClient(
        path=DB_DIR
    )

    chroma_collection = _reset_chroma_collection(
        db
    )

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    # ========================================================
    # 4. Document store
    # ========================================================

    docstore = SimpleDocumentStore()

    docstore.add_documents(
        nodes
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )

    # ========================================================
    # 5. Vector index
    # ========================================================

    print(
        "🧠 Creating vector index..."
    )

    VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # ========================================================
    # 6. BM25
    # ========================================================

    print(
        "🔍 Building BM25 index..."
    )

    tokenized_corpus = [
        tokenize_for_bm25(
            node.get_content()
        )
        for node in leaf_nodes
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    # ========================================================
    # 7. Persist artifacts
    # ========================================================

    _save_artifacts(
        bm25=bm25,
        nodes=nodes,
        leaf_nodes=leaf_nodes,
    )

    print(
        "\n✅ Ingestion complete!"
    )

    print(
        "📚 Chroma vectors persisted."
    )

    print(
        "🔍 BM25 index persisted."
    )

    print(
        "🧩 Node hierarchy persisted."
    )


# ============================================================
# Standalone
# ============================================================

if __name__ == "__main__":

    run_ingestion()