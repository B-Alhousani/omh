import pytest

import build_index
from build_index import index_chunks
from config import BATCH_SIZE


class FakeCollection:
    def __init__(self):
        self.calls = []

    def add(self, ids, embeddings, documents, metadatas):
        self.calls.append({
            "ids": ids,
            "embeddings": embeddings,
            "documents": documents,
            "metadatas": metadatas,
        })


@pytest.fixture
def fake_embed(monkeypatch):
    monkeypatch.setattr(
        build_index.ollama,
        "embed",
        lambda model, input: {"embeddings": [[0.0]] * len(input)},
    )


def make_chunks(count):
    return [
        {"source": "om-500.pdf", "page": i, "text": f"chunk {i}"}
        for i in range(count)
    ]


def test_writes_one_call_per_batch(fake_embed):
    collection = FakeCollection()
    list(index_chunks(collection, make_chunks(BATCH_SIZE + 5)))
    assert len(collection.calls) == 2
    assert len(collection.calls[0]["ids"]) == BATCH_SIZE
    assert len(collection.calls[1]["ids"]) == 5


def test_ids_are_unique_and_sequential_across_batches(fake_embed):
    collection = FakeCollection()
    list(index_chunks(collection, make_chunks(BATCH_SIZE + 5)))
    written = [chunk_id for call in collection.calls for chunk_id in call["ids"]]
    assert written == [str(i) for i in range(BATCH_SIZE + 5)]


def test_reports_cumulative_progress(fake_embed):
    collection = FakeCollection()
    progress = list(index_chunks(collection, make_chunks(BATCH_SIZE + 5)))
    assert progress == [BATCH_SIZE, BATCH_SIZE + 5]


def test_carries_source_and_page_into_metadata(fake_embed):
    collection = FakeCollection()
    list(index_chunks(collection, make_chunks(2)))
    assert collection.calls[0]["metadatas"] == [
        {"source": "om-500.pdf", "page": 0},
        {"source": "om-500.pdf", "page": 1},
    ]


def test_embeds_every_chunk_in_the_batch(fake_embed):
    collection = FakeCollection()
    list(index_chunks(collection, make_chunks(3)))
    assert len(collection.calls[0]["embeddings"]) == 3


def test_empty_input_writes_nothing(fake_embed):
    collection = FakeCollection()
    assert list(index_chunks(collection, [])) == []
    assert collection.calls == []
