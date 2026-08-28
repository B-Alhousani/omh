CHUNK_SIZE = 800
OVERLAP = 150


def chunk_page(page):
    """Split one cleaned page dict into chunk dicts."""
    text = page["text"]
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start:start + CHUNK_SIZE]
        chunks.append({
            "source": page["source"],
            "page": page["page"],
            "text": piece,
        })
        start += CHUNK_SIZE - OVERLAP
    return chunks


if __name__ == "__main__":
    from load_pdfs import load_pdfs
    from clean_text import clean_page

    all_chunks = []
    for p in load_pdfs():
        p["text"] = clean_page(p["text"])
        all_chunks.extend(chunk_page(p))

    print("Total chunks:", len(all_chunks))
    print("--- example ---")
    print(all_chunks[10]["source"], "page", all_chunks[10]["page"])
    print(all_chunks[10]["text"])