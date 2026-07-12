from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

from backend.app.core.config import get_settings
from backend.app.models.schemas import ChunkingConfig
from backend.app.services.chunking import FixedSizeTextChunker, RecursiveTextChunker
from backend.app.services.embeddings import get_embedding_service
from backend.app.services.generator import AnswerGenerator
from backend.app.services.parser import DocumentParser
from backend.app.services.retriever import HybridRetriever
from backend.app.services.vector_store import VectorStore


class RAGPipeline:
    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.parser = DocumentParser(
            settings.parsed_dir,
            backend=settings.mineru_backend,
            model_path=settings.mineru_model_path,
            device_map=settings.mineru_device_map,
            image_analysis=settings.mineru_image_analysis,
            pdf_dpi=settings.mineru_pdf_dpi,
        )
        self.embedder = get_embedding_service()
        self.vector_store = VectorStore()
        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator()
        self.chunk_records: list[dict] = self._load_chunk_records()
        if self.chunk_records:
            self.retriever.refresh_corpus(self.chunk_records)

    def ingest_files(
        self,
        files: list[Path],
        chunking_config: ChunkingConfig,
        progress_callback: Callable[[int, str, str], None] | None = None,
    ) -> tuple[list[str], list[str], int]:
        """Ingest files while rolling back any partially-created index state."""
        previous_records = list(self.chunk_records)
        existing_parsed_paths = set(self.settings.parsed_dir.glob("*.json"))
        try:
            return self._ingest_files(files, chunking_config, progress_callback)
        except Exception:
            created_paths = set(self.settings.parsed_dir.glob("*.json")) - existing_parsed_paths
            document_ids: list[str] = []
            for path in created_paths:
                try:
                    document_ids.append(json.loads(path.read_text(encoding="utf-8"))["document_id"])
                except (OSError, json.JSONDecodeError, KeyError):
                    pass

            try:
                self.vector_store.delete_by_document_ids(document_ids)
            except Exception:
                pass
            self.chunk_records = previous_records
            try:
                self._rewrite_chunk_records()
            except Exception:
                pass
            try:
                self.retriever.refresh_corpus(self.chunk_records)
            except Exception:
                pass
            for path in created_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def _ingest_files(
        self,
        files: list[Path],
        chunking_config: ChunkingConfig,
        progress_callback: Callable[[int, str, str], None] | None = None,
    ) -> tuple[list[str], list[str], int]:
        document_ids: list[str] = []
        filenames: list[str] = []
        new_chunks: list[dict] = []
        chunker = self._build_chunker(chunking_config)
        total_files = max(1, len(files))

        for file_index, file_path in enumerate(files, start=1):
            if progress_callback:
                progress_callback(
                    min(5 + int((file_index - 1) / total_files * 35), 40),
                    "parsing",
                    f"正在解析 {file_path.name}",
                )

            parsed = self._parse_file(file_path, file_index, total_files, progress_callback)
            document_ids.append(parsed.document_id)
            filenames.append(parsed.filename)

            if progress_callback:
                progress_callback(
                    min(45 + int(file_index / total_files * 15), 60),
                    "chunking",
                    f"正在切分 {parsed.filename}",
                )
            chunks = chunker.split_text(parsed.text)
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{parsed.document_id}-{idx}"
                metadata = {
                    "document_id": parsed.document_id,
                    "filename": parsed.filename,
                    "chunk_index": idx,
                    "source_path": parsed.metadata["source_path"],
                }
                new_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": parsed.document_id,
                        "content": chunk,
                        "metadata": metadata,
                    }
                )

        if progress_callback:
            progress_callback(75, "embedding", "正在生成向量")
        embeddings = self.embedder.embed_texts([item["content"] for item in new_chunks])
        if new_chunks:
            if progress_callback:
                progress_callback(90, "storing", "正在写入向量库")
            self.vector_store.upsert(
                ids=[item["chunk_id"] for item in new_chunks],
                embeddings=embeddings,
                documents=[item["content"] for item in new_chunks],
                metadatas=[item["metadata"] for item in new_chunks],
            )
        self.chunk_records.extend(new_chunks)
        self._persist_chunk_records()
        self.retriever.refresh_corpus(self.chunk_records)
        if progress_callback:
            progress_callback(100, "completed", "入库完成")
        return document_ids, filenames, len(new_chunks)

    async def ask(self, question: str, top_k: int) -> tuple[str, list[dict], str]:
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        contexts = [item.content for item in retrieved]
        answer, prompt = await self.generator.generate(question, contexts)
        payload = [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "score": item.score,
                "retrieval_sources": item.retrieval_sources,
                "content": item.content,
                "metadata": item.metadata,
            }
            for item in retrieved
        ]
        return answer, payload, prompt

    def list_documents(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        chunk_count_by_document: dict[str, int] = {}
        for record in self.chunk_records:
            chunk_count_by_document[record["document_id"]] = chunk_count_by_document.get(record["document_id"], 0) + 1

        for path in sorted(self.settings.parsed_dir.glob("*.json")):
            if path.name == self.settings.chunk_store_path.name:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            document_id = payload["document_id"]
            documents.append(
                {
                    "document_id": document_id,
                    "filename": payload["filename"],
                    "chunk_count": chunk_count_by_document.get(document_id, 0),
                    "source_path": metadata.get("source_path", ""),
                    "parser_backend": metadata.get("parser_backend", ""),
                }
            )
        documents.sort(key=lambda item: item["filename"])
        return documents

    def delete_document(self, document_id: str) -> dict[str, Any]:
        target_path = self.settings.parsed_dir / f"{document_id}.json"
        if not target_path.exists():
            raise FileNotFoundError(f"Document {document_id} was not found.")

        payload = json.loads(target_path.read_text(encoding="utf-8"))
        filename = payload["filename"]

        deleted_records = [record for record in self.chunk_records if record["document_id"] == document_id]
        deleted_chunk_ids = [record["chunk_id"] for record in deleted_records]
        deleted_chunk_count = len(deleted_records)

        self.vector_store.delete(deleted_chunk_ids)
        self.chunk_records = [record for record in self.chunk_records if record["document_id"] != document_id]
        self._rewrite_chunk_records()
        self.retriever.refresh_corpus(self.chunk_records)

        source_path = payload.get("metadata", {}).get("source_path")
        if source_path:
            source_file = Path(source_path)
            if source_file.exists():
                source_file.unlink()
        target_path.unlink()

        return {
            "document_id": document_id,
            "filename": filename,
            "deleted_chunk_count": deleted_chunk_count,
        }

    def _load_chunk_records(self) -> list[dict]:
        path = self.settings.chunk_store_path
        if not path.exists():
            return []
        records: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def _persist_chunk_records(self) -> None:
        self._write_chunk_records(self.chunk_records)

    def _write_chunk_records(self, records: list[dict]) -> None:
        path = self.settings.chunk_store_path
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        try:
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _rewrite_chunk_records(self) -> None:
        self._write_chunk_records(self.chunk_records)

    def _build_chunker(self, chunking_config: ChunkingConfig):
        if chunking_config.chunk_method == "fixed":
            return FixedSizeTextChunker(chunking_config)
        return RecursiveTextChunker(chunking_config)

    def _parse_file(
        self,
        file_path: Path,
        file_index: int,
        total_files: int,
        progress_callback: Callable[[int, str, str], None] | None,
    ):
        def page_progress(page_number: int, total_pages: int) -> None:
            if not progress_callback:
                return
            file_base_progress = int((file_index - 1) / total_files * 35)
            file_progress = int(page_number / max(1, total_pages) * (35 / total_files))
            progress_callback(
                min(5 + file_base_progress + file_progress, 40),
                "parsing",
                f"正在解析 {file_path.name} 第 {page_number}/{total_pages} 页",
            )

        return self.parser._parse_with_local_model(file_path, page_progress)
