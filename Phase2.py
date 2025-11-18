from sentence_transformers import SentenceTransformer
import chromadb
import json
import os

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "julius_caesar_scenes_clean"
DB_PATH = "chroma_db_scenes_clean"
DATA_PATH = "julius_caesar_scene_chunks_CLEAN.jsonl"


def load_chunks(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input file: {path}")

    docs, metas, ids = [], [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            doc_id = f"act_{chunk['act']}_scene_{chunk['scene']}"

            ids.append(doc_id)
            docs.append(chunk["text"])
            metas.append({"act": chunk["act"], "scene": chunk["scene"]})

    print(f"Loaded {len(docs)} scene chunks.")
    return docs, metas, ids


def build_db(docs, metas, ids):
    print("Loading embedder...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Creating persistent Chroma DB → {DB_PATH}")
    # Use PersistentClient instead of Client with Settings
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("Encoding documents...")
    embeddings = model.encode(docs, convert_to_numpy=True)

    print("Upserting...")
    collection.upsert(
        ids=ids,
        documents=docs,
        embeddings=embeddings.tolist(),  # Convert to list for ChromaDB
        metadatas=metas
    )

    print("✔ DB saved.")
    return collection


def main():
    docs, metas, ids = load_chunks(DATA_PATH)
    build_db(docs, metas, ids)

    print("Verifying DB...")
    # Use PersistentClient for verification too
    client = chromadb.PersistentClient(path=DB_PATH)

    collections = client.list_collections()
    print(f"Collections: {[c.name for c in collections]}")
    print("Phase2 completed successfully.")


if __name__ == "__main__":
    main()
