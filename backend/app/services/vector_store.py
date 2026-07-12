from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.app.core.config import get_settings


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=settings.chroma_collection_name)

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def dense_search(self, query_embedding: list[float], top_k: int) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    def delete_by_document_ids(self, document_ids: list[str]) -> None:
        if document_ids:
            self.collection.delete(where={"document_id": {"$in": document_ids}})
