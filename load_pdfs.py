from pathlib import Path
from pypdf import PdfReader

DATA_DIR = Path("data")


def load_pdfs():
    
    pages = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({
                "source": pdf_path.name,
                "page": page_num,
                "text": text,
            })
    return pages


if __name__ == "__main__":
    pages = load_pdfs()
    print("Total pages:", len(pages))
    first = pages[0]
    print("=== ", first["source"], "page", first["page"], "===")
    print(first["text"])