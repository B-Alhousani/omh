import chromadb
import ollama

from config import BATCH_SIZE, COLLECTION, DB_DIR, EMBED_MODEL
from load_pdfs import load_pdfs
from clean_text import clean_page
from chunk_text import chunk_page


def build_chunks():
    chunks = []
    for page in load_pdfs():
        page["text"] = clean_page(page["text"])
        chunks.extend(chunk_page(page))
    return chunks


def fresh_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    return client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def index_chunks(collection, chunks):
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        response = ollama.embed(model=EMBED_MODEL, input=[c["text"] for c in batch])
        collection.add(
            ids=[str(start + offset) for offset in range(len(batch))],
            embeddings=response["embeddings"],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
        )
        yield start + len(batch)


def build():
    chunks = build_chunks()
    collection = fresh_collection()
    for done in index_chunks(collection, chunks):
        print(f"\r{done}/{len(chunks)}", end="")
    print("\nDone.")


if __name__ == "__main__":
    build()
