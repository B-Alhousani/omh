import pytest
from rank_bm25 import BM25Okapi

import search as search_module
from search import Retriever, tokenize


@pytest.fixture
def chunks():
    return {
        "ids": ["0", "1", "2"],
        "documents": ["email attachments", "internet browsing", "shared ipad login"],
        "metadatas": [
            {"source": "om-505.pdf", "page": 1},
            {"source": "om-500.pdf", "page": 2},
            {"source": "pc-522.pdf", "page": 3},
        ],
    }


@pytest.fixture
def bm25(chunks):
    return BM25Okapi([tokenize(d) for d in chunks["documents"]])


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return self.scores[:len(pairs)]


class FakeCollection:
    def __init__(self, order):
        self.order = order

    def query(self, query_embeddings, n_results):
        return {"ids": [self.order]}


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("E-Mail Policy, OM-505!") == ["e", "mail", "policy", "om", "505"]


def test_tokenize_keeps_digits():
    assert tokenize("Section 3 of 7") == ["section", "3", "of", "7"]


def test_tokenize_empty_string():
    assert tokenize("") == []


def test_top_candidates_ranks_agreement_first(chunks):
    retriever = Retriever(None, chunks, None, None)
    ranks = {"0": 2, "1": 0, "2": 1}
    out = retriever._top_candidates(ranks, ranks)
    assert [c["source"] for c in out] == ["om-500.pdf", "pc-522.pdf", "om-505.pdf"]


def test_top_candidates_rewards_consistency_over_one_strong_vote(chunks):
    embedding_rank = {"0": 0, "1": 5, "2": 9}
    keyword_rank = {"0": 9, "1": 0, "2": 5}
    out = Retriever(None, chunks, None, None)._top_candidates(embedding_rank, keyword_rank)
    assert [c["page"] for c in out] == [2, 1, 3]


def test_top_candidates_carries_metadata(chunks):
    ranks = {"0": 0, "1": 1, "2": 2}
    out = Retriever(None, chunks, None, None)._top_candidates(ranks, ranks)
    assert out[0] == {"text": "email attachments", "source": "om-505.pdf", "page": 1}


def test_rank_by_keyword_puts_exact_match_first(chunks, bm25):
    ranks = Retriever(None, chunks, bm25, None)._rank_by_keyword("ipad")
    assert ranks["2"] == 0


def test_rank_by_keyword_ranks_every_chunk(chunks, bm25):
    ranks = Retriever(None, chunks, bm25, None)._rank_by_keyword("email")
    assert sorted(ranks.values()) == [0, 1, 2]


def test_rerank_orders_by_score_and_truncates_to_k():
    reranker = FakeReranker([0.1, 0.9, 0.5])
    candidates = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    out = Retriever(None, None, None, reranker)._rerank("q", candidates, k=2)
    assert [c["text"] for c in out] == ["b", "c"]


def test_rerank_pairs_the_query_with_each_candidate():
    reranker = FakeReranker([1.0, 1.0])
    Retriever(None, None, None, reranker)._rerank(
        "gmail?", [{"text": "a"}, {"text": "b"}], k=2
    )
    assert reranker.pairs == [("gmail?", "a"), ("gmail?", "b")]


def test_search_runs_all_four_stages(chunks, bm25, monkeypatch):
    monkeypatch.setattr(
        search_module.ollama,
        "embed",
        lambda model, input: {"embeddings": [[0.0]]},
    )
    retriever = Retriever(
        FakeCollection(["2", "1", "0"]), chunks, bm25, FakeReranker([0.1, 0.2, 0.3])
    )
    out = retriever.search("shared ipad", k=2)
    assert len(out) == 2
    assert all({"text", "source", "page"} == set(hit) for hit in out)


def test_search_never_returns_more_than_k(chunks, bm25, monkeypatch):
    monkeypatch.setattr(
        search_module.ollama,
        "embed",
        lambda model, input: {"embeddings": [[0.0]]},
    )
    retriever = Retriever(
        FakeCollection(["0", "1", "2"]), chunks, bm25, FakeReranker([0.5, 0.4, 0.3])
    )
    assert len(retriever.search("email", k=1)) == 1
