"""
Phase 4: Prompt Engineering & Generation with Gemini API

Features:
- Multiple API key rotation with rate limiting
- Detailed system prompt for Shakespearean Scholar persona
- 3-5 second cooldown between API calls
- Source citation enforcement
"""

import os
import time
import random
import requests

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


app = FastAPI(title="Julius Caesar RAG API with Generation")


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


# ======== API Key Management ========
class APIKeyManager:
    """Manages multiple Gemini API keys with rotation and cooldown."""

    def __init__(self, keys: List[str], cooldown_min: float = 6.0, cooldown_max: float = 8.0):
        if not keys:
            raise ValueError("At least one API key is required")
        self.keys = keys
        self.current_index = 0
        self.cooldown_min = cooldown_min
        self.cooldown_max = cooldown_max
        self.last_call_time = 0
        print(f" Initialized API Key Manager with {len(keys)} key(s)")

    def get_next_key(self) -> str:
        """Get the next API key in rotation."""
        key = self.keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.keys)
        return key

    def apply_cooldown(self):
        """Apply cooldown between API calls."""
        elapsed = time.time() - self.last_call_time
        cooldown = random.uniform(self.cooldown_min, self.cooldown_max)

        if elapsed < cooldown:
            sleep_time = cooldown - elapsed
            print(f" Cooldown: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_call_time = time.time()


# ======== Configuration ========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db_scenes_clean")
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", DEFAULT_CHROMA_DIR)
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

# Load multiple API keys from environment
# Format: GEMINI_API_KEYS="key1,key2,key3" or individual GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
api_keys_str = os.environ.get(
    "GEMINI_API_KEYS", '*')
if api_keys_str:
    API_KEYS = [k.strip() for k in api_keys_str.split(",") if k.strip()]
else:
    # Try loading individual keys
    API_KEYS = []
    i = 1
    while True:
        key = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip()
        if not key:
            break
        API_KEYS.append(key)
        i += 1

    # Fallback to single key
    if not API_KEYS:
        single_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if single_key:
            API_KEYS = [single_key]

if not API_KEYS:
    raise RuntimeError(
        "No Gemini API keys found. Set one of:\n"
        "  - GEMINI_API_KEYS='key1,key2,key3'\n"
        "  - GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.\n"
        "  - GEMINI_API_KEY"
    )

api_key_manager = APIKeyManager(API_KEYS, cooldown_min=6.0, cooldown_max=7.0)


# ======== System Prompt ========
SYSTEM_PROMPT = """You are a highly accurate literary analysis assistant specialized in 
Shakespeare's *Julius Caesar*.  
Your primary task is to answer questions ONLY using the information found 
in the retrieved context chunks from the play and its authoritative summaries and answer in the SHORTEST and CLEANEST WAY POSSIBLE. 
Your goal is to provide responses that are factually correct, text-grounded, 
concise, and aligned with the canonical interpretation of the play.

STRICT GUIDELINES:

1. Grounding:
   - STRICTLY Use ONLY the retrieved context to answer.
   - If the retrieved context does not contain the answer, STRICTLY say:
     "The retrieved passages do not contain enough information to answer this 
     confidently."
   - STRICTLY Never fabricate scenes, events, interpretations, or quotes.

2. Accuracy:
   - STRICTLY Treat the retrieved text as the single source of truth.
   - STRICTLY Follow Shakespeare's canonical plot, character motivations, themes, and events.
   - If multiple retrieved chunks disagree, STRICTLY prioritize the one that comes from:
       (a) The play text itself  
       (b) Reputable scholarly summaries of Shakespeare  
   - Do not invent analysis beyond what is text-supported.

3. Style and Length(STRICLY FOLLOW):
   - STRICTLY Answer clearly, directly, and WITHOUT archaic language imitation.
   - Keep responses SHORT to LESS THAN 50 characters per answer and CRISP—typically 1-2 sentences maximum.
   - For factual questions (who, what, when, where), provide ONLY the essential 
     information without elaboration.
   - For analytical questions (why, how, analyze), provide a CONCISE explanation 
     in 2-3 sentences focusing on the key insight.
   - STRICTLY NEVER write lengthy paragraphs or over-explain.

4. Quotes:
   - DO NOT quote if the retrieved chunk contains the exact line.
   - Never invent or paraphrase a quote as if it were original.

5. Multi-part or choice questions:
   - If asked "Which of the following…?", select only the answer that the retrieved 
     context supports.
   - For "All of the above" questions, choose it ONLY if each individual item is 
     supported in the retrieved text.
   - Answer with just the choice (e.g., "All of the above") without unnecessary 
     explanation unless specifically asked.

6. Ambiguity:
   - If the retrieved chunks give partial information, synthesize faithfully
     but STRICTLY do NOT fill gaps creatively.
   - Keep uncertain responses small: state what IS known, then note what's missing.

7. Identity:
   - You are not a character, reenactor, or dramatizer.
   - You are an expert literary assistant providing text-grounded answers.


OUTPUT REQUIREMENTS:
- STRICTLY Provide a single, direct answer.
- Keep responses SHORT—most answers should be 1-2 sentences and STRICTLY keep MAXIMUM LENGTH OF 50 characters.
- Never mention retrieval, embeddings, or technical system details.
- STRICTLY Do not add unnecessary preambles like "According to the text..." or "Based on 
  the retrieved context..."
- Get straight to the answer.
"""


# ======== Load Embedder ========
print(f"Loading embedding model: {EMBED_MODEL_NAME}")
embedder = SentenceTransformer(EMBED_MODEL_NAME)


# ======== Initialize ChromaDB ========
if not os.path.exists(CHROMA_DB_DIR):
    raise RuntimeError(
        f"Chroma DB directory not found at {CHROMA_DB_DIR}\n"
        "Run Phase2 first to build the vector database."
    )

try:
    print(f" Loading Chroma DB from: {CHROMA_DB_DIR}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collections = chroma_client.list_collections()

    if not collections:
        raise RuntimeError(f"No collections found in {CHROMA_DB_DIR}")

    collection_name = collections[0].name
    collection = chroma_client.get_collection(collection_name)
    print(f" Loaded collection: {collection_name}")

except Exception as e:
    raise RuntimeError(f"Failed to initialize ChromaDB: {e}")


# ======== Utility Functions ========
def embed_text(texts: List[str]) -> np.ndarray:
    """Generate embeddings for input texts."""
    embs = embedder.encode(texts, convert_to_numpy=True)
    if embs.ndim == 1:
        embs = np.expand_dims(embs, 0)
    return embs


def retrieve_with_chroma(query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top-k relevant passages from ChromaDB."""
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
        docs.append({
            "document": doc,
            "metadata": meta or {},
            "distance": dist,
        })
    print("=============================Retrieved data========================")
    print(doc)
    print("=========================================================================")

    return docs


def format_context(retrieved: List[Dict[str, Any]]) -> str:
    """Format retrieved passages into context string."""
    if not retrieved:
        return "No relevant context found."

    context_parts = []
    for i, doc in enumerate(retrieved, 1):
        meta = doc["metadata"]
        act = meta.get("act", "?")
        scene = meta.get("scene", "?")
        text = doc["document"]

        context_parts.append(
            f"[Passage {i} - Act {act}, Scene {scene}]\n{text}\n"
        )

    return "\n".join(context_parts)


def generate_answer_with_gemini(query: str, context: str) -> str:
    """Generate answer using OpenRouter API with key rotation and cooldown."""

    # Apply cooldown before API call
    api_key_manager.apply_cooldown()

    # Get next API key
    current_key = api_key_manager.get_next_key()

    # OpenRouter headers
    headers = {
        "Authorization": f"Bearer {current_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "JuliusCaesar-RAG"
    }

    # Choose model
    model_name = "meta-llama/llama-3.1-8b-instruct"
    # Alternative models:
    # "qwen/qwen2.5-7b-instruct"
    # "mistral/mistral-small-latest"
    # "anthropic/claude-3-haiku"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Based on the following passages from Julius Caesar, answer the question.

**Question:** {query}

**Context:**
{context}

Remember: Only use the provided context and cite Act and Scene."""
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1500
    }

    try:
        print(
            f" Generating answer with OpenRouter (key #{api_key_manager.current_index})")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers
        )

        if response.status_code != 200:
            print(" OpenRouter Error:", response.text)
            return f"Error: OpenRouter returned {response.status_code}"

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        print("Error calling OpenRouter:", e)
        return f"Error generating response: {str(e)}"


# ======== FastAPI Endpoint ========
@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest):
    """Main RAG endpoint with retrieval and generation."""
    q = body.query.strip()
    k = body.k or 5

    if not q:
        raise HTTPException(status_code=400, detail="Query text is empty")

    print(f"\n Query: {q}")

    # Step 1: Embed query
    q_emb = embed_text([q])

    # Step 2: Retrieve relevant passages
    retrieved = retrieve_with_chroma(q_emb, k)
    print(f" Retrieved {len(retrieved)} passages")

    # Step 3: Format context
    context = format_context(retrieved)

    # Step 4: Generate answer with Gemini
    answer = generate_answer_with_gemini(q, context)

    # Step 5: Prepare sources
    sources = [
        Source(chunk=r["document"], metadata=r["metadata"])
        for r in retrieved
    ]

    return QueryResponse(answer=answer, sources=sources)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "api_keys_loaded": len(API_KEYS),
        "collection": collection.name if collection else None
    }


# ======== Run Server ========
if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print(" Julius Caesar RAG API with Gemini Generation")
    print("="*60)
    print(f" API Keys loaded: {len(API_KEYS)}")
    print(
        f" Cooldown: {api_key_manager.cooldown_min}-{api_key_manager.cooldown_max}s")
    print(f" Server starting at http://127.0.0.1:8002")
    print("="*60 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=8002)
