import re

import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import CANDIDATES, COLLECTION, DB_DIR, EMBED_MODEL, RERANK_MODEL, TOP_K


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self, collection, chunks, bm25, reranker):
        self.collection = collection
        self.chunks = chunks
        self.bm25 = bm25
        self.reranker = reranker

    def search(self, query, k=TOP_K):
        embedding_rank = self._rank_by_embedding(query)
        keyword_rank = self._rank_by_keyword(query)
        candidates = self._top_candidates(embedding_rank, keyword_rank)
        return self._rerank(query, candidates, k)

    def _rank_by_embedding(self, query):
        response = ollama.embed(model=EMBED_MODEL, input=query)
        hits = self.collection.query(
            query_embeddings=response["embeddings"], n_results=len(self.chunks["ids"])
        )
        return {chunk_id: rank for rank, chunk_id in enumerate(hits["ids"][0])}

    def _rank_by_keyword(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return {self.chunks["ids"][i]: rank for rank, i in enumerate(order)}

    def _top_candidates(self, embedding_rank, keyword_rank):
        def rrf_score(chunk_id):
            return 1 / (10 + embedding_rank[chunk_id]) + 1 / (10 + keyword_rank[chunk_id])

        position = {chunk_id: i for i, chunk_id in enumerate(self.chunks["ids"])}
        best_ids = sorted(self.chunks["ids"], key=rrf_score, reverse=True)[:CANDIDATES]
        candidates = []
        for chunk_id in best_ids:
            i = position[chunk_id]
            candidates.append({
                "text": self.chunks["documents"][i],
                "source": self.chunks["metadatas"][i]["source"],
                "page": self.chunks["metadatas"][i]["page"],
            })
        return candidates

    def _rerank(self, query, candidates, k):
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in order[:k]]


def load_retriever():
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    chunks = collection.get()
    bm25 = BM25Okapi([tokenize(d) for d in chunks["documents"]])
    reranker = CrossEncoder(RERANK_MODEL)
    return Retriever(collection, chunks, bm25, reranker)
