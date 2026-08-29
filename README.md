# OMH Policy RAG

Ask questions about four OMH Official Policy Manual PDFs and get answers grounded
in them, with source documents and page numbers. Runs locally through Ollama.

## Setup

Install [Ollama](https://ollama.com/download), then:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Python 3.9+:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Run

Build the index once (`chroma_db/` is not committed):

```bash
python build_index.py
```

Then ask:

```bash
python ask.py                      # interactive session
python ask.py "your question"      # single question
```

First run downloads the reranker model (~90 MB) from Hugging Face.

## Tests

```bash
python -m pytest tests/
```

## How it works

`load_pdfs.py` → `clean_text.py` → `chunk_text.py` → `build_index.py` embeds
chunks into Chroma. `search.py` retrieves in four stages: embedding search, BM25
keyword search, Reciprocal Rank Fusion, then a cross-encoder rerank. `ask.py`
prompts `llama3.2` with the top chunks.

Sources are built from chunk metadata, not model output, so they always reflect
the documents actually retrieved.

Settings live in `config.py`.
