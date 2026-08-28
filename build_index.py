import chromadb
import ollama

from load_pdfs import load_pdfs
from clean_text import clean_page
from chunk_text import chunk_page

DB_DIR = "chroma_db"
COLLECTION = "omh_policies"


def build():
    client = chromadb.PersistentClient(path=DB_DIR)
    # delete old collection so re-runs start fresh
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION)

    chunks = []
    for p in load_pdfs():
        p["text"] = clean_page(p["text"])
        chunks.extend(chunk_page(p))

    for i, ch in enumerate(chunks):
        emb = ollama.embeddings(model="nomic-embed-text", prompt=ch["text"])
        col.add(
            ids=[str(i)],
            embeddings=[emb["embedding"]],
            documents=[ch["text"]],
            metadatas=[{"source": ch["source"], "page": ch["page"]}],
        )
        print(f"\r{i + 1}/{len(chunks)}", end="")
    print("\nDone.")


if __name__ == "__main__":
    build()