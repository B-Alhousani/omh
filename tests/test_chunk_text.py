from chunk_text import chunk_page
from config import CHUNK_SIZE, OVERLAP


def make_page(text):
    return {"source": "om-505.pdf", "page": 4, "text": text}


def test_short_page_is_one_chunk():
    chunks = chunk_page(make_page("short text"))
    assert len(chunks) == 1
    assert chunks[0]["text"] == "short text"


def test_long_page_splits_on_stride():
    chunks = chunk_page(make_page("x" * (CHUNK_SIZE * 2)))
    stride = CHUNK_SIZE - OVERLAP
    assert len(chunks) == len(range(0, CHUNK_SIZE * 2, stride))


def test_consecutive_chunks_overlap():
    text = "".join(str(i % 10) for i in range(CHUNK_SIZE * 2))
    chunks = chunk_page(make_page(text))
    assert chunks[0]["text"][-OVERLAP:] == chunks[1]["text"][:OVERLAP]


def test_every_chunk_keeps_source_and_page():
    chunks = chunk_page(make_page("y" * (CHUNK_SIZE * 3)))
    assert all(c["source"] == "om-505.pdf" and c["page"] == 4 for c in chunks)


def test_empty_page_yields_no_chunks():
    assert chunk_page(make_page("")) == []
