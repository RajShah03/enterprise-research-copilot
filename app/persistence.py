import json
import pickle
import shutil
from pathlib import Path

import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import (
    ChromaVectorStore,
)

from app.config import embed_model


# ============================================================
# Runtime Storage
# ============================================================

RUNTIME_DIR = Path("./runtime")

UPLOADS_DIR = RUNTIME_DIR / "uploads"
CHROMA_DIR = RUNTIME_DIR / "chroma_db"
STORAGE_DIR = RUNTIME_DIR / "storage"


# ============================================================
# Document Paths
# ============================================================

def get_document_paths(
    document_id: str,
):
    workspace_dir = (
        UPLOADS_DIR / document_id
    )

    artifact_dir = (
        STORAGE_DIR / document_id
    )

    chroma_dir = (
        CHROMA_DIR / document_id
    )

    return {
        "workspace_dir": workspace_dir,
        "artifact_dir": artifact_dir,
        "chroma_dir": chroma_dir,
        "bm25_path": (
            artifact_dir / "bm25.pkl"
        ),
        "nodes_path": (
            artifact_dir / "nodes.pkl"
        ),
        "metadata_path": (
            artifact_dir / "metadata.json"
        ),
        "collection_name": (
            f"financial_docs_{document_id}"
        ),
    }


# ============================================================
# Save Document Metadata
# ============================================================

def save_document_metadata(
    document_id: str,
    filename: str,
    leaf_node_count: int,
):
    paths = get_document_paths(
        document_id
    )

    paths["artifact_dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "document_id": document_id,
        "filename": filename,
        "leaf_node_count": leaf_node_count,
    }

    with paths["metadata_path"].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


# ============================================================
# Load Document Metadata
# ============================================================

def load_document_metadata(
    document_id: str,
):
    paths = get_document_paths(
        document_id
    )

    if not paths["metadata_path"].exists():
        raise FileNotFoundError(
            f"Document metadata not found for "
            f"'{document_id}'."
        )

    with paths["metadata_path"].open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# Cleanup Document
# ============================================================

def cleanup_document(
    document_id: str,
):
    paths = get_document_paths(
        document_id
    )

    for directory in (
        paths["workspace_dir"],
        paths["artifact_dir"],
        paths["chroma_dir"],
    ):
        if directory.exists():
            shutil.rmtree(
                directory,
                ignore_errors=True,
            )


# ============================================================
# Load RAG Resources
# ============================================================

def load_rag_resources(
    document_id: str,
):
    """
    Loads the already-built RAG resources for one document.

    No document ingestion or embedding happens here.
    """

    paths = get_document_paths(
        document_id
    )

    if not paths["bm25_path"].exists():
        raise FileNotFoundError(
            f"BM25 artifact not found for "
            f"document '{document_id}'."
        )

    if not paths["nodes_path"].exists():
        raise FileNotFoundError(
            f"Node artifact not found for "
            f"document '{document_id}'."
        )

    if not paths["chroma_dir"].exists():
        raise FileNotFoundError(
            f"Chroma database not found for "
            f"document '{document_id}'."
        )

    print(
        f"🗄️ Loading Chroma resources for "
        f"document: {document_id}"
    )

    db = chromadb.PersistentClient(
        path=str(
            paths["chroma_dir"]
        )
    )

    try:
        chroma_collection = (
            db.get_collection(
                paths["collection_name"]
            )
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"Chroma collection "
            f"'{paths['collection_name']}' "
            f"was not found."
        ) from exc

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    with paths["bm25_path"].open(
        "rb"
    ) as file:
        bm25 = pickle.load(file)

    with paths["nodes_path"].open(
        "rb"
    ) as file:
        node_artifacts = pickle.load(file)

    leaf_nodes = node_artifacts[
        "leaf_nodes"
    ]

    print(
        f"✅ Loaded "
        f"{len(leaf_nodes)} "
        f"leaf nodes."
    )

    return (
        index,
        bm25,
        leaf_nodes,
    )