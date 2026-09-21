import re
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)
from pydantic import BaseModel
from flashrank import Ranker

from app.agent import CorrectiveRAGAgent
from app.ingestion import ingest_pdf
from app.persistence import (
    cleanup_document,
    get_document_paths,
    load_document_metadata,
    load_rag_resources,
)
from app.retriever import HybridRetriever


# ============================================================
# Configuration
# ============================================================

MAX_PDF_SIZE = 25 * 1024 * 1024

DOCUMENT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{32}$"
)


# ============================================================
# Application Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    print(
        "\n🚀 Starting Enterprise Research Copilot API..."
    )

    print(
        "⚡ Loading shared FlashRank reranker..."
    )

    app.state.reranker = Ranker(
        model_name="ms-marco-MiniLM-L-12-v2"
    )

    app.state.document_agents = {}

    print(
        "✅ Enterprise Research Copilot ready."
    )

    yield

    print(
        "\n🛑 Shutting down Enterprise Research Copilot..."
    )

    app.state.document_agents.clear()
    app.state.reranker = None


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="Enterprise Research Copilot API",
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request Models
# ============================================================

class QueryRequest(
    BaseModel
):
    document_id: str
    query: str


# ============================================================
# Helpers
# ============================================================

def _validate_document_id(
    document_id: str,
):
    if not DOCUMENT_ID_PATTERN.fullmatch(
        document_id
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID.",
        )


def _get_or_load_agent(
    app: FastAPI,
    document_id: str,
):
    agents = app.state.document_agents

    if document_id in agents:
        return agents[document_id]

    try:
        (
            index,
            bm25,
            leaf_nodes,
        ) = load_rag_resources(
            document_id
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                "The uploaded document is no longer "
                "available. Please upload it again."
            ),
        ) from exc

    retriever = HybridRetriever(
        index=index,
        bm25=bm25,
        leaf_nodes=leaf_nodes,
        reranker=app.state.reranker,
    )

    agent = CorrectiveRAGAgent(
        hybrid_retriever=retriever,
    )

    agents[document_id] = agent

    return agent


# ============================================================
# Health Check
# ============================================================

@app.get("/")
def health_check():

    return {
        "status": "ok",
        "service": "enterprise-research-copilot",
    }


# ============================================================
# Upload PDF
# ============================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile,
):
    filename = Path(
        file.filename or "document.pdf"
    ).name

    if not filename.lower().endswith(
        ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    document_id = uuid4().hex

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

    destination = (
        workspace_dir / filename
    )

    try:
        total_size = 0

        with destination.open(
            "wb"
        ) as output:

            while True:
                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(
                    chunk
                )

                if (
                    total_size
                    > MAX_PDF_SIZE
                ):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "PDF exceeds the 25 MB "
                            "maximum file size."
                        ),
                    )

                output.write(
                    chunk
                )

        await file.close()

        # ----------------------------------------------------
        # Basic PDF signature validation
        # ----------------------------------------------------

        with destination.open(
            "rb"
        ) as input_file:

            signature = (
                input_file.read(5)
            )

        if signature != b"%PDF-":
            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded file is not a valid PDF."
                ),
            )

        print(
            f"\n📄 Received PDF: {filename}"
        )

        print(
            f"🆔 Document ID: {document_id}"
        )

        print(
            "🧠 Building document-specific RAG resources..."
        )

        ingestion_result = ingest_pdf(
            source_pdf=destination,
            document_id=document_id,
        )

        # ----------------------------------------------------
        # Load resources and cache the agent
        # ----------------------------------------------------

        agent = _get_or_load_agent(
            app,
            document_id,
        )

        metadata = load_document_metadata(
            document_id
        )

        return {
            "status": "success",
            "document_id": document_id,
            "filename": metadata[
                "filename"
            ],
            "leaf_node_count": (
                ingestion_result[
                    "leaf_node_count"
                ]
            ),
        }

    except HTTPException:
        cleanup_document(
            document_id
        )
        raise

    except Exception as exc:
        cleanup_document(
            document_id
        )

        print(
            f"❌ PDF processing failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process the uploaded PDF."
            ),
        ) from exc


# ============================================================
# Query Uploaded Document
# ============================================================

@app.post("/query")
def process_query(
    request: Request,
    payload: QueryRequest,
):
    query = payload.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    _validate_document_id(
        payload.document_id
    )

    rag_agent = _get_or_load_agent(
        request.app,
        payload.document_id,
    )

    try:
        result = rag_agent.run(
            query
        )

        return {
            "status": "success",
            "query": query,
            "document_id": payload.document_id,
            **result,
        }

    except Exception as exc:
        print(
            f"❌ Query processing failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Query processing failed.",
        ) from exc