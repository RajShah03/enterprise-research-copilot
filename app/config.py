import os

from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ============================================================
# Environment
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing from the .env file."
    )

if not GROQ_MODEL:
    raise ValueError(
        "GROQ_MODEL is missing from the .env file. "
        "Add a currently supported Groq model."
    )


# ============================================================
# LLM
# ============================================================

llm = Groq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.1,
)


# ============================================================
# Embedding Model
# ============================================================

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)