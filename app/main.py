from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.persistence import load_rag_resources
from app.retriever import HybridRetriever
from app.agent import CorrectiveRAGAgent


# ============================================================
# Application Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Loads the already-built RAG resources before the API
    starts accepting requests.

    No document ingestion happens here.
    """

    print(
        "\n🚀 Starting Enterprise Research Copilot API..."
    )

    print(
        "📦 Loading persisted RAG resources..."
    )

    index, bm25, leaf_nodes = (
        load_rag_resources()
    )

    print(
        "⚡ Initializing hybrid retriever..."
    )

    retriever = HybridRetriever(
        index=index,
        bm25=bm25,
        leaf_nodes=leaf_nodes,
    )

    print(
        "🤖 Initializing RAG agent..."
    )

    app.state.rag_agent = CorrectiveRAGAgent(
        hybrid_retriever=retriever,
    )

    print(
        "✅ Enterprise Research Copilot ready."
    )

    yield

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    print(
        "\n🛑 Shutting down Enterprise Research Copilot..."
    )

    app.state.rag_agent = None


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="Enterprise Research Copilot API",
    version="1.2.0",
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
# Request Model
# ============================================================

class QueryRequest(
    BaseModel
):
    query: str


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
# Query
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

    rag_agent = getattr(
        request.app.state,
        "rag_agent",
        None,
    )

    if rag_agent is None:

        raise HTTPException(
            status_code=503,
            detail="RAG service is not ready.",
        )

    try:

        result = rag_agent.run(
            query
        )

        return {
            "status": "success",
            "query": query,
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