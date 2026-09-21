import pickle
import re
import shutil
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
from app.persistence import (
    get_document_paths,
    save_document_metadata,
)


# ============================================================
# BM25 Tokenizer
# ============================================================

def tokenize_for_bm25(
    text: str,
):
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
# Document Ingestion
# ============================================================

def ingest_pdf(
    source_pdf: Path,
    document_id: str,
):
    """
    Creates an isolated RAG workspace for one PDF.

    Returns:
        document_id
        filename
        leaf_node_count
    """

    source_pdf = Path(
        source_pdf
    ).resolve()

    if not source_pdf.exists():
        raise FileNotFoundError(
            f"PDF not found: {source_pdf}"
        )

    if source_pdf.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    paths = get_document_paths(
        document_id
    )

    workspace_dir = (
        paths["workspace_dir"]
    )

    workspace_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_dir = (
        paths["artifact_dir"]
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chroma_dir = (
        paths["chroma_dir"]
    )

    chroma_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Copy source PDF into its isolated workspace
    # --------------------------------------------------------

    destination_pdf = (
    workspace_dir /
    source_pdf.name
    )

    if (
        source_pdf.resolve()
        != destination_pdf.resolve()
    ):
        shutil.copy2(
            source_pdf,
            destination_pdf,
        )

    print(
        f"\n📥 Ingesting document: "
        f"{source_pdf.name}"
    )

    # --------------------------------------------------------
    # Load document
    # --------------------------------------------------------

    reader = SimpleDirectoryReader(
        str(workspace_dir)
    )

    documents = reader.load_data()

    if not documents:
        raise ValueError(
            "No readable content found in the PDF."
        )

    print(
        f"📄 Loaded {len(documents)} document(s)."
    )

    # --------------------------------------------------------
    # Hierarchical nodes
    # --------------------------------------------------------

    print(
        "🧩 Creating hierarchical nodes..."
    )

    node_parser = (
        HierarchicalNodeParser.from_defaults(
            chunk_sizes=[
                1024,
                128,
            ]
        )
    )

    nodes = (
        node_parser.get_nodes_from_documents(
            documents
        )
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

    # --------------------------------------------------------
    # ChromaDB
    # --------------------------------------------------------

    print(
        "🗄️ Creating document-specific ChromaDB..."
    )

    db = chromadb.PersistentClient(
        path=str(chroma_dir)
    )

    chroma_collection = (
        db.create_collection(
            paths["collection_name"]
        )
    )

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    # --------------------------------------------------------
    # Docstore + Vector Index
    # --------------------------------------------------------

    docstore = SimpleDocumentStore()

    docstore.add_documents(
        nodes
    )

    storage_context = (
        StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore,
        )
    )

    print(
        "🧠 Creating vector index..."
    )

    VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

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

    with paths["bm25_path"].open(
        "wb"
    ) as file:
        pickle.dump(
            bm25,
            file,
        )

    with paths["nodes_path"].open(
        "wb"
    ) as file:
        pickle.dump(
            {
                "nodes": nodes,
                "leaf_nodes": leaf_nodes,
            },
            file,
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    save_document_metadata(
        document_id=document_id,
        filename=source_pdf.name,
        leaf_node_count=len(
            leaf_nodes
        ),
    )

    print(
        "\n✅ Document ingestion complete."
    )

    print(
        f"📄 Document: {source_pdf.name}"
    )

    print(
        f"🔹 Leaf nodes: {len(leaf_nodes)}"
    )

    print(
        f"🆔 Document ID: {document_id}"
    )

    return {
        "document_id": document_id,
        "filename": source_pdf.name,
        "leaf_node_count": len(
            leaf_nodes
        ),
    }