# Enterprise Research Copilot

A full-stack, document-grounded financial research assistant built with **Next.js, FastAPI, LlamaIndex, ChromaDB, BM25, FlashRank, Hugging Face embeddings, and Groq**.

The system combines dense semantic retrieval and lexical retrieval, reranks retrieved candidates, evaluates whether the available evidence is sufficient, and can perform corrective retrieval before generating a grounded answer with source citations.

---

## Overview

Enterprise Research Copilot is designed to answer questions using evidence retrieved from a predefined research document rather than relying solely on the language model's general knowledge.

The application follows a **Corrective RAG (CRAG)** workflow:

1. Retrieve candidate evidence using vector search and BM25.
2. Combine and deduplicate the retrieved candidates.
3. Rerank candidates using FlashRank.
4. Build the final evidence context.
5. Evaluate the retrieved evidence as `COMPLETE`, `PARTIAL`, or `NONE`.
6. If the evidence is classified as `NONE`, rewrite the query and perform corrective retrieval.
7. Generate a grounded answer using the Groq LLM.
8. Return the answer together with source metadata and page references.

The project separates **document ingestion** from **application serving**, allowing the API to load already-built retrieval resources instead of rebuilding the knowledge base whenever the server starts.

---

## Current Scope

The current application uses a **predefined NVIDIA financial research PDF** as its source document.

The complete ingestion, hybrid retrieval, reranking, Corrective RAG, answer generation, citation, and frontend workflows were developed and tested using this document.

The current project is intended as a portfolio-scale AI engineering application demonstrating an end-to-end **full-stack Corrective RAG system**.

---

## Architecture

```text
                         ┌──────────────────────┐
                         │      Next.js UI      │
                         │   Research Interface │
                         └──────────┬───────────┘
                                    │
                              HTTP / JSON
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    FastAPI Backend   │
                         │      POST /query     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Corrective RAG Agent│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Hybrid Retriever   │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
             Vector Retrieval                    BM25
             ChromaDB / LlamaIndex             Lexical Search
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                         Candidate Pool + Dedup
                                    │
                                    ▼
                              FlashRank
                              Reranking
                                    │
                                    ▼
                           Top Retrieved Passages
                                    │
                                    ▼
                         Evidence Evaluation
                       COMPLETE / PARTIAL / NONE
                                    │
                              ┌─────┴─────┐
                              │           │
                         Sufficient      NONE
                              │           │
                              │      Query Rewrite
                              │           │
                              │      Re-retrieval
                              │           │
                              └─────┬─────┘
                                    ▼
                              Groq LLM
                                    │
                                    ▼
                         Grounded Answer + Citations
                                    │
                                    ▼
                             Next.js Interface
```

---

## Ingestion Architecture

Documents are processed separately from the serving application.

```text
Source Document
      │
      ▼
SimpleDirectoryReader
      │
      ▼
HierarchicalNodeParser
      │
      ├── Parent Nodes
      │
      └── Leaf Nodes
             │
        ┌────┴─────┐
        │          │
        ▼          ▼
   BGE Embedding   BM25 Tokenization
        │          │
        ▼          ▼
     ChromaDB    BM25 Index
```

The current hierarchical configuration uses:

```text
Parent chunk size: 1024
Leaf chunk size:   128
```

The system indexes the leaf nodes for retrieval while preserving the hierarchical node structure as persisted artifacts.

> Note: the current serving pipeline does not perform parent-node expansion. Hierarchical nodes are created and persisted, but retrieved leaf passages are directly used as the final evidence context.

---

## Core Features

### Hybrid Retrieval

The retriever combines two complementary retrieval approaches:

* **Dense vector retrieval** using LlamaIndex and ChromaDB
* **Lexical retrieval** using BM25

The current retrieval configuration is:

```text
Vector candidates: 15
BM25 candidates:   10
Final passages:     6
```

The candidate sets are combined, exact duplicate passages are removed, and the resulting candidates are reranked.

---

### FlashRank Reranking

FlashRank is used as a second-stage reranker after candidate retrieval.

Current reranker:

```text
ms-marco-MiniLM-L-12-v2
```

The purpose of reranking is to improve the relevance of the final passages passed to the evidence evaluation and answer-generation stages.

---

### Corrective RAG

The system evaluates the complete retrieved context using three evidence states:

```text
COMPLETE
PARTIAL
NONE
```

When the initial evidence is classified as `NONE`, the system performs corrective retrieval:

```text
Original Query
      ↓
Query Rewriting
      ↓
Corrective Retrieval
      ↓
Evidence Evaluation
      ↓
Answer / Not Found
```

Corrective retrieval is not triggered when the initial retrieved context already contains useful evidence.

---

### Grounded Answer Generation

The answer-generation stage instructs the LLM to:

* use only retrieved evidence
* preserve financial figures and units
* avoid unsupported facts
* distinguish between different financial metrics
* avoid unsupported rankings
* identify missing information
* avoid unsupported future or causal claims
* cite the retrieved sources used for the answer

---

### Source Citations

Answers use citations such as:

```text
NVIDIA's Data Center revenue increased by 68%. [SOURCE 1]
```

The frontend converts source references into clickable citation indicators.

Selecting a citation takes the user directly to the corresponding source card.

Each source includes:

* source number
* document name
* page number when available
* relevant retrieved passage

---

### Persistent Knowledge Base

The ingestion process creates persistent retrieval resources:

```text
chroma_db/

storage/
├── bm25.pkl
└── nodes.pkl
```

The serving application loads these resources instead of rebuilding the complete knowledge base whenever the API starts.

---

### Full-Stack Interface

The Next.js frontend provides:

* research question input
* example questions
* loading state
* error handling
* grounded answer display
* clickable source citations
* document and page references
* copy-answer action
* evidence status
* optional technical details

The frontend acts as the presentation layer while the FastAPI application contains the RAG and retrieval logic.

---

## Technology Stack

| Layer                     | Technology                               |
| ------------------------- | ---------------------------------------- |
| Frontend                  | Next.js, React, TypeScript, Tailwind CSS |
| Backend                   | FastAPI, Python                          |
| RAG Framework             | LlamaIndex                               |
| Vector Database           | ChromaDB                                 |
| Lexical Retrieval         | rank-bm25                                |
| Reranking                 | FlashRank                                |
| Embeddings                | BAAI/bge-small-en-v1.5                   |
| LLM                       | Groq                                     |
| Document Loading          | LlamaIndex SimpleDirectoryReader         |
| Chunking                  | HierarchicalNodeParser                   |
| API Validation            | Pydantic                                 |
| Environment Configuration | python-dotenv                            |

---

## Project Structure

```text
enterprise-research-copilot/
│
├── app/
│   ├── agent.py
│   ├── config.py
│   ├── ingestion.py
│   ├── main.py
│   ├── persistence.py
│   └── retriever.py
│
├── data/
│   └── README.md
│
├── frontend/
│   ├── app/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── public/
│   ├── package.json
│   └── ...
│
├── evaluation_runner.py
├── ingest.py
├── nvidia_rag_evaluation_dataset_v2.json
├── requirements.txt
├── run.py
├── README.md
└── .gitignore
```

Generated, sensitive, or locally stored files are intentionally excluded from Git:

```text
.env
venv/
chroma_db/
storage/
data/*.pdf
frontend/node_modules/
frontend/.next/
evaluation_results.json
evaluation_results.csv
evaluation_summary.json
```

---

## Prerequisites

Install:

* Python 3.12+
* Node.js and npm
* Git

A Groq API key is required for answer generation.

---

## Backend Setup

From the project root:

```powershell
cd "D:\enterprise research copilot"
```

Create and activate a virtual environment if one does not already exist:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install backend dependencies:

```powershell
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_supported_groq_model
```

The `.env` file is intentionally excluded from version control.

---

## Add the Source Document

Place the research PDF used by the application inside:

```text
data/
```

The current demonstration uses the NVIDIA financial research PDF used during development and testing.

The source PDF is intentionally excluded from the Git repository unless redistribution rights have been verified.

---

## Build the RAG Knowledge Base

After adding or changing the source document, run:

```powershell
python ingest.py
```

The ingestion process:

```text
Loads document
    ↓
Creates hierarchical nodes
    ↓
Creates embeddings
    ↓
Stores vectors in ChromaDB
    ↓
Builds BM25 index
    ↓
Persists BM25 and node artifacts
```

After ingestion, the generated resources are stored locally in:

```text
chroma_db/
storage/
```

You do not need to run ingestion every time the API starts.

Run ingestion again only when the indexed source document changes or the retrieval resources need to be rebuilt.

---

## Start the Backend

From the project root:

```powershell
python run.py
```

The FastAPI application runs locally on:

```text
http://127.0.0.1:8000
```

### Health Check

```text
GET /
```

Example response:

```json
{
  "status": "ok",
  "service": "enterprise-research-copilot"
}
```

### Query Endpoint

```text
POST /query
```

Example request:

```json
{
  "query": "What were NVIDIA's Data Center revenues in fiscal 2026?"
}
```

The response contains:

* query
* answer
* evidence status
* sources
* Corrective RAG status
* rewritten query when corrective retrieval is triggered
* source metadata

---

## Frontend Setup

Open another terminal:

```powershell
cd "D:\enterprise research copilot\frontend"
```

Install frontend dependencies:

```powershell
npm install
```

Create:

```text
frontend/.env.local
```

with:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Start the development server:

```powershell
npm run dev
```

Open:

```text
http://localhost:3000
```

The frontend communicates with the FastAPI backend through the `/query` endpoint.

---

## Run the Complete Application

Two processes are required during local development.

### Terminal 1 — Backend

```powershell
cd "D:\enterprise research copilot"
.\venv\Scripts\Activate.ps1
python run.py
```

### Terminal 2 — Frontend

```powershell
cd "D:\enterprise research copilot\frontend"
npm run dev
```

Then open:

```text
http://localhost:3000
```

---

## Example Questions

The current NVIDIA financial research corpus can be queried with questions such as:

```text
What were NVIDIA's Data Center revenues in fiscal 2026?
```

```text
How did NVIDIA's Data Center revenue change year over year?
```

```text
What risks did NVIDIA identify in its annual review?
```

The frontend displays the generated answer together with supporting source passages and page references.

---

## Evidence Handling

The application distinguishes between three evidence states.

### Complete

The retrieved evidence supports the major requirements of the question.

### Partial

Useful evidence exists, but one or more requested facts, metrics, or relationships are not supported by the retrieved evidence.

### None

The retrieved evidence does not provide useful support for the question.

When evidence is insufficient, the answer-generation logic is designed to avoid presenting unsupported information as fact.

---

## Corrective Retrieval

Corrective retrieval is triggered only when the initial context evaluation returns:

```text
NONE
```

The original query is then rewritten into a retrieval-focused query while preserving the user's intent, requested metrics, company names, dates, and comparison requirements.

The rewritten query is sent through the same hybrid retrieval and reranking pipeline.

---

## Design Decisions

### Why RAG instead of fine-tuning?

The purpose of the application is to answer questions from document-specific information.

RAG retrieves relevant evidence at query time without requiring the document knowledge to be embedded directly into the language model's parameters.

### Why vector search + BM25?

Dense vector retrieval helps identify semantically related passages, while BM25 is useful for exact terms, numbers, names, and financial terminology.

Using both retrieval methods provides complementary candidate retrieval before reranking.

### Why FlashRank?

The first retrieval stage collects a broader set of candidates. FlashRank then reranks those candidates so that the strongest passages can be selected for the final context.

### Why ChromaDB?

ChromaDB provides persistent local vector storage and integrates with the LlamaIndex vector-store layer used by this project.

### Why Groq?

Groq is used as the LLM inference provider through the LlamaIndex Groq integration.

### Why separate ingestion and serving?

Document processing and application serving have different responsibilities.

The ingestion process creates the retrieval resources, while the serving application loads those resources and handles user queries.

This prevents the API from rebuilding the knowledge base whenever it starts.

---

## Limitations

This project is a portfolio-scale AI research application rather than a fully hardened enterprise deployment.

Current limitations include:

* The current demonstration is based on a predefined NVIDIA financial research PDF.
* The complete application has been developed and tested using this source document; it has not been independently validated across other document collections.
* Ingestion rebuilds the Chroma collection rather than performing incremental document updates.
* ChromaDB is currently used as a local persistent vector database.
* BM25 and node artifacts are persisted locally using Python pickle files.
* The application does not include authentication or authorization.
* No multi-user conversation or chat-history system is implemented.
* No document-upload interface is included.
* Citation formatting is normalized, but the application does not independently verify semantic support for every generated claim.
* Current retrieval configuration values such as `15 / 10 / 6` are engineering choices for this implementation and are not claimed to be universally optimal.
* Hierarchical nodes are persisted, but parent expansion is not currently part of the serving pipeline.

---

## Development and Evaluation

The repository includes evaluation-related files used during development:

```text
evaluation_runner.py
nvidia_rag_evaluation_dataset_v2.json
```

These files document the development and testing process for the current system.

The evaluation tooling should not be interpreted as a definitive benchmark of general-purpose RAG performance.

The application was primarily validated through end-to-end testing of:

* document ingestion
* vector retrieval
* BM25 retrieval
* reranking
* evidence evaluation
* corrective retrieval
* grounded answer generation
* source citations
* FastAPI integration
* Next.js frontend integration

---

## Production Build

Before deployment, verify the frontend production build:

```powershell
cd "D:\enterprise research copilot\frontend"
npm run build
```

Verify the Python backend compiles successfully:

```powershell
cd "D:\enterprise research copilot"
python -m compileall app ingest.py run.py
```

---

## Security and Repository Hygiene

The following files are intentionally excluded from version control:

```text
.env
venv/
chroma_db/
storage/
data/*.pdf
frontend/node_modules/
frontend/.next/
evaluation_results.json
evaluation_results.csv
evaluation_summary.json
```

The `.env` file contains credentials and must never be committed.

The generated ChromaDB, BM25, and node artifacts are local serving resources rather than source-code files.

The source PDF should only be redistributed publicly after verifying the applicable rights.

---

## Deployment Considerations

The application consists of two layers:

```text
Next.js Frontend
       ↓
FastAPI Backend
       ↓
CRAG Retrieval Pipeline
```

The frontend and backend can be deployed as separate services.

The deployed backend also needs access to the required persistent retrieval resources:

```text
chroma_db/
storage/
```

For a production deployment, these local resources would typically be replaced or hosted using deployment-appropriate persistent storage and infrastructure.

The current application should therefore be considered **deployment-ready as a portfolio project after configuring its persistent retrieval resources**, rather than as a fully production-hardened enterprise system.

---

## Current Project Status

The project currently provides a complete full-stack CRAG-based financial research application:

```text
Next.js Frontend
       ↓
FastAPI Backend
       ↓
Hybrid Vector + BM25 Retrieval
       ↓
FlashRank Reranking
       ↓
Evidence Evaluation
       ↓
Corrective Retrieval when required
       ↓
Groq Answer Generation
       ↓
Citations + Source Evidence
```

The current implementation represents the completed project scope.

No additional advanced RAG components are required for the current project milestone.

---

## License

## License

No open-source license has been applied to this repository.
All rights are reserved by the author unless otherwise stated.

Any third-party documents used with the application may be subject to their own copyright and usage restrictions.

---

## Author

**Raj Shah**

AI/ML Engineer

GitHub: https://github.com/RajShah03

LinkedIn: https://www.linkedin.com/in/raj22shah/
