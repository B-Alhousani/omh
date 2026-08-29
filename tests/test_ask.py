import pytest

import ask as ask_module
from ask import NOT_FOUND, ask, format_sources


class FakeRetriever:
    def search(self, query, k):
        return [{"text": "email rules", "source": "om-505.pdf", "page": 1}]


@pytest.fixture
def reply(monkeypatch):
    def _reply(content):
        monkeypatch.setattr(
            ask_module.ollama,
            "chat",
            lambda model, messages: {"message": {"content": content}},
        )

    return _reply


def test_refusal_returns_no_sources(reply):
    reply(NOT_FOUND)
    answer, hits = ask("What is the capital of France", FakeRetriever())
    assert hits == []


def test_grounded_answer_keeps_its_sources(reply):
    reply("Staff may not use personal email. [E-Mail Policy (OM-505)]")
    answer, hits = ask("Can staff use Gmail?", FakeRetriever())
    assert len(hits) == 1


def test_refusal_is_detected_despite_trailing_punctuation(reply):
    reply("I could not find this in the provided policies")
    answer, hits = ask("anything", FakeRetriever())
    assert hits == []


def test_groups_pages_under_one_document():
    hits = [
        {"source": "om-505.pdf", "page": 3},
        {"source": "om-505.pdf", "page": 1},
    ]
    assert format_sources(hits) == "- E-Mail Policy (OM-505) - pages 1, 3"


def test_single_page_uses_singular_label():
    hits = [{"source": "om-500.pdf", "page": 2}]
    assert format_sources(hits) == "- Acceptable Use of Internet Policy (OM-500) - page 2"


def test_deduplicates_repeated_pages():
    hits = [{"source": "pc-522.pdf", "page": 4}, {"source": "pc-522.pdf", "page": 4}]
    assert format_sources(hits) == (
        "- Staff and Client Use of Shared iPads (PC-522) - page 4"
    )


def test_documents_are_listed_in_a_stable_order():
    hits = [
        {"source": "pc-522.pdf", "page": 1},
        {"source": "om-500.pdf", "page": 1},
        {"source": "i-100intranet.pdf", "page": 1},
    ]
    assert format_sources(hits) == (
        "- Preface (I-100) - page 1\n"
        "- Acceptable Use of Internet Policy (OM-500) - page 1\n"
        "- Staff and Client Use of Shared iPads (PC-522) - page 1"
    )
