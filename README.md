# Evidentia

**Evidence-Grounded Document Intelligence**

A full-stack **RAG/CRAG application** that lets users upload any PDF and chat with it using **hybrid retrieval, FlashRank reranking, evidence evaluation, corrective retrieval, and Groq-powered generation**.

## Features

* Upload and chat with any PDF
* Document-specific retrieval using `document_id`
* Hierarchical document chunking
* Dense semantic retrieval with ChromaDB
* BM25 lexical retrieval
* Hybrid retrieval + FlashRank reranking
* Evidence evaluation: `COMPLETE`, `PARTIAL`, `NONE`
* Query rewriting and corrective re-retrieval when evidence is insufficient
* Grounded Markdown answers
* Expandable source evidence
* Expandable retrieval/debug details
* Light / dark mode
* FastAPI backend + Next.js frontend

## Architecture

```text
User
 ↓
Next.js Frontend
 ↓
POST /upload
 ↓
PDF Parsing
 ↓
Hierarchical Chunking
 ↓
ChromaDB + BM25
 ↓
POST /query
 ↓
Hybrid Retrieval
 ↓
FlashRank Reranking
 ↓
Evidence Evaluation
 ↓
Corrective Retrieval (when NONE)
 ↓
Groq LLM
 ↓
Grounded Answer
 ↓
Expandable Sources
```

Each uploaded PDF gets its own `document_id` and isolated ChromaDB, BM25, and node artifacts.

## Tech Stack

| Layer             | Technology                               |
| ----------------- | ---------------------------------------- |
| Frontend          | Next.js, React, TypeScript, Tailwind CSS |
| Backend           | Python, FastAPI                          |
| RAG               | LlamaIndex                               |
| Vector DB         | ChromaDB                                 |
| Lexical Retrieval | BM25                                     |
| Reranking         | FlashRank                                |
| Embeddings        | BAAI/bge-small-en-v1.5                   |
| LLM               | Groq                                     |
| Markdown          | react-markdown                           |

## Project Structure

```text
evidentia/
│
├── app/
│   ├── agent.py
│   ├── config.py
│   ├── ingestion.py
│   ├── main.py
│   ├── persistence.py
│   └── retriever.py
│
├── frontend/
│   ├── app/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── package.json
│   └── package-lock.json
│
├── evaluation_runner.py
├── nvidia_rag_evaluation_dataset_v2.json
├── requirements.txt
├── run.py
├── README.md
└── .gitignore
```

Generated runtime data is stored under `runtime/` and excluded from Git.

## Setup

### Backend

```powershell
cd "D:\enterprise research copilot"

python -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

Create `.env`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_supported_groq_model
```

Start the API:

```powershell
python run.py
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

### Frontend

```powershell
cd "D:\enterprise research copilot\frontend"

npm install
npm run dev
```

Frontend:

```text
http://localhost:3000
```

## Usage

1. Upload a PDF.
2. Wait for document processing and indexing.
3. Ask questions about the uploaded document.
4. View the grounded answer.
5. Expand **Sources** to inspect the retrieved evidence.

Example questions:

```text
What are the key financial highlights?

What factors affected revenue?

What risks are mentioned in the document?

Summarize the main findings.
```

## Corrective RAG

The system evaluates retrieved evidence before generation:

```text
COMPLETE → Generate answer
PARTIAL  → Generate with limitations
NONE     → Rewrite query → Re-retrieve → Evaluate again
```

The model is instructed to answer only from retrieved evidence and to explicitly identify unsupported information.

## Runtime Storage

```text
runtime/
├── uploads/
├── chroma_db/
└── storage/
```

The runtime directory contains uploaded PDFs and generated retrieval artifacts and is excluded from version control.

## Limitations

* PDF-only input
* 25 MB upload limit
* Local persistent storage
* No authentication or multi-user system
* No server-side chat history
* Current evaluation dataset is based on the original NVIDIA test corpus
* Not hardened as a production multi-user enterprise service

## Verification

Backend syntax check:

```powershell
python -m compileall app run.py
```

Frontend production build:

```powershell
cd frontend
npm run build
```

## License

No open-source license has been applied to this repository. All rights are reserved by the author unless otherwise stated.

Any third-party documents used with the application may be subject to their own copyright and usage restrictions.

## Author

**Raj Shah**

GitHub: https://github.com/RajShah03
LinkedIn: https://www.linkedin.com/in/raj22shah/

```
```
