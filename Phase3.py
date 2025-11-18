"""
Phase3: RAG API Backend (Updated for latest ChromaDB)

This version works with ChromaDB >= 0.5.0
- Uses PersistentClient (no deprecated Settings)
- Loads persistent vector DB created in Phase2
"""

import chromadb
from typing import List, Dict, Any, Optional
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer


app = FastAPI(title="RAG API")


# ======== Request & Response Models ========
class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5


class Source(BaseModel):
    chunk: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


# ======== PATH CONFIG ========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db_scenes_clean")
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", DEFAULT_CHROMA_DIR)


# ======== Load Embedder ========
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
embedder = SentenceTransformer(EMBED_MODEL_NAME)


# ======== Initialize Chroma Client ========

if not os.path.exists(CHROMA_DB_DIR):
    raise RuntimeError(
        f"Chroma DB directory not found at {CHROMA_DB_DIR}\n"
        "Did you run Phase2 successfully?"
    )

try:
    print(f"🔄 Loading Chroma DB from: {CHROMA_DB_DIR}")

    # Use PersistentClient instead of deprecated Client(Settings(...))
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    collections = chroma_client.list_collections()
    if not collections:
        raise RuntimeError(
            f"No collections found inside Chroma DB at {CHROMA_DB_DIR}"
        )

    print("Found collections:", [c.name for c in collections])

    # Load first (and only) collection
    collection_name = collections[0].name
    collection = chroma_client.get_collection(collection_name)

    print(f"✔ Loaded Chroma collection: {collection_name}")

except Exception as e:
    raise RuntimeError(
        "\n❌ Failed to initialize ChromaDB:\n"
        + str(e)
        + "\n\nFix:\n"
        " - Ensure Phase2 ran and created the DB folder\n"
        " - Ensure Chroma is installed: pip install chromadb\n"
        " - Ensure CHROMA_DB_DIR points to your DB\n"
    )


# ======== Utility Functions ========
def embed_text(texts: List[str]) -> np.ndarray:
    embs = embedder.encode(texts, convert_to_numpy=True)
    if embs.ndim == 1:
        embs = np.expand_dims(embs, 0)
    return embs


def retrieve_with_chroma(query_embedding: np.ndarray, k: int = 5):
    """Run vector similarity search using chroma."""
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    docs_list = results.get("documents", [[]])
    meta_list = results.get("metadatas", [[]])
    dist_list = results.get("distances", [[]])

    for doc, meta, dist in zip(docs_list[0], meta_list[0], dist_list[0]):
        docs.append(
            {
                "document": doc,
                "metadata": meta or {},
                "distance": dist,
            }
        )

    return docs


def generate_answer(query: str, retrieved: List[Dict[str, Any]]) -> str:
    """Simple extractive placeholder."""
    if not retrieved:
        return "I found no relevant information in the knowledge base."

    answer_chunks = "\n\n".join([f"- {r['document']}" for r in retrieved])

    return (
        "SYNTHESIZED ANSWER (extractive):\n"
        "Relevant chunks found:\n\n" + answer_chunks
    )


# ======== FastAPI Endpoint ========
@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest):
    q = body.query
    k = body.k or 5

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query text is empty")

    q_emb = embed_text([q])
    retrieved = retrieve_with_chroma(q_emb, k)

    sources = [
        {"chunk": r["document"], "metadata": r["metadata"]} for r in retrieved
    ]

    answer = generate_answer(q, retrieved)

    return {"answer": answer, "sources": sources}


# ======== Run Server ========
if __name__ == "__main__":
    import uvicorn

    print("\n🚀 RAG API is running at http://127.0.0.1:8000/query\n")
    uvicorn.run(app, host="127.0.0.1", port=8001)
