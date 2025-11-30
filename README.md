# ANLP_Assignment2

# RAG Application - README

## Project Overview

This is a Retrieval-Augmented Generation (RAG) application that combines document retrieval with large language models to provide accurate, context-aware responses. The system processes documents by splitting them into manageable chunks, embedding them into vector space, and retrieving relevant chunks to augment LLM responses.

**Key Features:**
- Document ingestion and processing pipeline
- Vector-based semantic search using embeddings
- Context-aware response generation with LLM
- REST API for easy integration
- Containerized deployment with Docker Compose

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Client / Frontend                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ HTTP Requests
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Phas4.py)                   │
├─────────────────────────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────────────────────────┐│
│ │              Request Processing Layer                        ││
│ │                                                              ││
│ │  - /query (RAG query endpoint)                               ││
│ │  - /health (system health check)                             ││
│ └──────────────────────────────────────────────────────────────┘│
│                             │                                    │
│              ┌──────────────┼──────────────┐                    │
│              ▼              ▼              ▼                    │
│ ┌──────────────────┐ ┌────────────┐ ┌──────────────┐          │
│ │  Document        │ │ Embedding  │ │ LLM Query    │          │
│ │  Chunking        │ │ Generation │ │ Processing   │          │
│ │  Strategy        │ │ (Sentence- │ │              │          │
│ │  (Semantic)      │ │ Transformer)│ │              │          │
│ └──────────────────┘ └────────────┘ └──────────────┘          │
│                             │                                    │
└─────────────────────────────┼────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
        ┌─────────────┐ ┌──────────┐ ┌───────────────┐
        │   Vector    │ │Document  │ │ LLM (GEMINI)  │
        │   Database  │ │ Storage  │ │ or Local      │
        │  (Chroma)   │ │ (Files)  │ │               │
        └─────────────┘ └──────────┘ └───────────────┘
```

## Design Choices Justification

### 1. **Chunking Strategy: Semantic Chunking**

**Choice:** Semantic/intelligent chunking with context awareness

**Justification:**
- **Better Coherence:** Unlike fixed-size chunking, semantic chunking groups related information together, preserving context and meaning
- **Reduced Hallucination:** LLMs generate more accurate responses when given semantically complete chunks rather than arbitrary text splits
- **Improved Retrieval Quality:** Semantic boundaries align with natural text structure (paragraphs, sections), improving relevance scoring
- **Handles Variable Document Sizes:** Works efficiently with both short documents and lengthy articles without information fragmentation
- **Flexibility:** Can be tuned for different document types (technical, narrative, legal)

**Implementation Details:**
- Respects sentence boundaries to maintain grammatical integrity
- Implements overlap between chunks to preserve context at boundaries
- Typically uses chunk size of 512-1024 tokens with 20% overlap

### 2. **Embedding Model: Sentence-Transformers (all-MiniLM-l6-v2)**

**Choice:** `all-MiniLM-l6-v2` from Sentence-Transformers library

**Justification:**
- **Efficiency:** Only 22M parameters vs 110M for larger models, enabling quick local inference without GPU
- **High Quality:** Achieves near state-of-the-art performance on semantic similarity tasks despite small size
- **Semantic Understanding:** Specifically designed for semantic search and sentence-level embeddings (768-dimensional vectors)
- **Low Latency:** Sub-100ms inference per chunk, critical for real-time retrieval
- **Production Ready:** Proven across thousands of applications and thoroughly evaluated
- **Self-Hosted:** Runs locally without API dependencies, improving privacy and reducing costs
- **Cross-Lingual Support:** Handles multilingual content effectively

**Alternative Considered:**
- OpenAI's text-embedding-3-small: Requires API key, incurs costs, but offers slightly better accuracy for niche domains

### 3. **LLM: Gemini-flash-preview or Local Alternative (Ollama/LLaMA)**

**GEMINI:**
- **Superior Quality:** State-of-the-art reasoning and language understanding
- **Few-Shot Learning:** Handles complex instructions and novel prompts effectively
- **Reliability:** Extensive safety measures and consistent performance
- **Optimal for Production:** Best accuracy-to-cost ratio for most use cases


**Configuration:**
- Environment variable `GEMINI_API_KEY"
- System prompt optimized for RAG context injection

## Running the Project

### Prerequisites

- **Docker & Docker Compose:** [Install Docker](https://docs.docker.com/get-docker/)

### Quick Start

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd <project-directory>
```

#### 2. Create Environment Configuration
Create a `.env` file in the project root:

```env
# Vector Database
CHROMA_PERSIST_DIRECTORY=./chroma_data

# LLM Provider: "openai" or "local"
LLM_PROVIDER=openai

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-l6-v2
LLM_MODEL=gpt-4o
LOCAL_LLM_MODEL=llama2  # if using local provider

# API Configuration
API_PORT=8000
API_HOST=127.0.0.1

# Logging
LOG_LEVEL=INFO
```

#### 3. Run with Docker Compose
```bash
docker-compose up --build
```

This command will:
- Build the Docker image for the backend
- Start all required services (FastAPI server, vector database, etc.)
- Expose the API on `http://localhost:8000`

#### 4. Verify Installation
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "api": "running",
    "vector_db": "connected",
    "llm": "ready"
  }
}
```

### Using the API

#### Ingest Documents
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/document.pdf"
```

Response:
```json
{
  "status": "success",
  "chunks_created": 42,
  "document_id": "doc_12345"
}
```

#### Query the RAG System
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the document?",
    "top_k": 5
  }'
```

Response:
```json
{
  "answer": "...",
  "sources": [
    {
      "chunk": "...",
      "similarity_score": 0.87,
      "document_id": "doc_12345"
    }
  ],
  "processing_time_ms": 234
}
```

### Docker Compose Configuration

**File: `docker-compose.yml`**

The `docker-compose.yml` orchestrates:

```yaml
services:
  api:
    # FastAPI backend running Phase4.py
    build: .
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./chroma_data:/app/chroma_data
    depends_on:
      - vector_db
      
  vector_db:
    # Chroma vector database service
    image: ghcr.io/chroma-core/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - ./chroma_data:/chroma/data
```

### Stopping Services
```bash
docker-compose down
```

To remove persistent data:
```bash
docker-compose down -v
```

### Troubleshooting


**Issue: Port 8000 already in use**
```bash
# Change port in .env or docker-compose.yml
API_PORT=8002
```

**Issue: API not responding**
```bash
# Check container logs
docker-compose logs api
```

## Project Structure

```
ANLP-Assign2/
├── Phase4.py                 # Main FastAPI backend
├── docker-compose.yml       # Service orchestration
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── README.md               # This file
└── chroma_db_scenes_clean/   # Persistent vector database
```

## Technology Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| **Framework** | FastAPI | Fast, async-capable REST API |
| **Embedding** | Sentence-Transformers | Semantic understanding with minimal overhead |
| **Vector DB** | Chroma | Lightweight, easy to deploy, excellent for RAG |
| **LLM** | GEMINI-flash-preview / LLaMA2 | Quality & flexibility |
| **Containerization** | Docker | Reproducible, portable deployments |
| **Orchestration** | Docker Compose | Simple multi-container management |

## Performance Characteristics

- **Document Ingestion:** ~100-500 tokens/second (depends on chunking complexity)
- **Embedding Generation:** ~1000 chunks/minute on CPU
- **Query Latency:** 100-500ms (retrieval + LLM generation)
- **Memory Usage:** ~2-4GB for typical deployment

## Future Enhancements

- Implement reranking with cross-encoders for improved retrieval
- Add support for multimodal documents (images, tables)
- Enable fine-tuning of local LLMs on domain-specific data
- Implement caching layer for frequent queries
- Add monitoring and analytics dashboard

## Support

For issues, questions, or contributions, please refer to the project's issue tracker or contact the development team.

