from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.core.config import get_settings
from backend.app.services.embeddings import get_embedding_service
from backend.app.services.vector_store import VectorStore


@dataclass
class RetrievedItem:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict
    score: float
    retrieval_sources: list[str]


class HybridRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.vector_store = VectorStore()
        self.chunk_records: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self._cross_encoder = None

    def refresh_corpus(self, chunk_records: list[dict]) -> None:
        self.chunk_records = chunk_records
        tokenized = [record["content"].split() for record in chunk_records]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        corpus = [record["content"] for record in chunk_records]
        if corpus:
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        else:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

    def retrieve(self, question: str, top_k: int) -> list[RetrievedItem]:
        rankings = []
        rankings.append(self._dense_search(question))
        rankings.append(self._bm25_search(question))
        rankings.append(self._sparse_search(question))
        fused = self._rrf(rankings)
        reranked = self._rerank(question, fused)
        return reranked[:top_k]

    def _dense_search(self, question: str) -> list[RetrievedItem]:
        if not self.chunk_records:
            return []
        query_embedding = self.embedding_service.embed_query(question)
        result = self.vector_store.dense_search(query_embedding, self.settings.dense_top_k)
        items: list[RetrievedItem] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            items.append(
                RetrievedItem(
                    chunk_id=chunk_id,
                    document_id=meta["document_id"],
                    content=doc,
                    metadata=meta,
                    score=float(1.0 / (1.0 + dist)),
                    retrieval_sources=["dense"],
                )
            )
        return items

    def _bm25_search(self, question: str) -> list[RetrievedItem]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(question.split())
        top_indices = np.argsort(scores)[::-1][: self.settings.bm25_top_k]
        return [
            RetrievedItem(
                chunk_id=self.chunk_records[idx]["chunk_id"],
                document_id=self.chunk_records[idx]["document_id"],
                content=self.chunk_records[idx]["content"],
                metadata=self.chunk_records[idx]["metadata"],
                score=float(scores[idx]),
                retrieval_sources=["bm25"],
            )
            for idx in top_indices
            if scores[idx] > 0
        ]

    def _sparse_search(self, question: str) -> list[RetrievedItem]:
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        query_vec = self.tfidf_vectorizer.transform([question])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][: self.settings.sparse_top_k]
        return [
            RetrievedItem(
                chunk_id=self.chunk_records[idx]["chunk_id"],
                document_id=self.chunk_records[idx]["document_id"],
                content=self.chunk_records[idx]["content"],
                metadata=self.chunk_records[idx]["metadata"],
                score=float(scores[idx]),
                retrieval_sources=["sparse"],
            )
            for idx in top_indices
            if scores[idx] > 0
        ]

    def _rrf(self, rankings: list[list[RetrievedItem]]) -> list[RetrievedItem]:
        scored: dict[str, float] = defaultdict(float)
        merged: dict[str, RetrievedItem] = {}
        for ranking in rankings:
            for rank, item in enumerate(ranking, start=1):
                scored[item.chunk_id] += 1.0 / (self.settings.rrf_k + rank)
                if item.chunk_id not in merged:
                    merged[item.chunk_id] = item
                else:
                    merged[item.chunk_id].retrieval_sources = sorted(
                        set(merged[item.chunk_id].retrieval_sources + item.retrieval_sources)
                    )
        ordered = sorted(merged.values(), key=lambda item: scored[item.chunk_id], reverse=True)
        for item in ordered:
            item.score = float(scored[item.chunk_id])
        return ordered

    def _rerank(self, question: str, candidates: list[RetrievedItem]) -> list[RetrievedItem]:
        if not candidates:
            return []
        try:
            from sentence_transformers import CrossEncoder

            if self._cross_encoder is None:
                self._cross_encoder = CrossEncoder(
                    self.settings.reranker_model_name,
                    device=self.settings.reranker_device,
                )
            pairs = [[question, item.content] for item in candidates]
            scores = self._cross_encoder.predict(pairs)
            for item, score in zip(candidates, scores):
                item.score = float(score)
            return sorted(candidates, key=lambda item: item.score, reverse=True)
        except Exception:
            return candidates
