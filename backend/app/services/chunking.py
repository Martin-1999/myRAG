from __future__ import annotations

from dataclasses import dataclass

from backend.app.models.schemas import ChunkingConfig


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict


class RecursiveTextChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def split_text(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        return self._split_with_separators(text, self.config.separators)

    def _split_with_separators(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.config.chunk_size:
            return [text]
        if not separators:
            return self._merge_small_pieces(list(text))

        separator = separators[0]
        if separator == "":
            return self._merge_small_pieces(list(text))

        pieces = text.split(separator)
        if len(pieces) == 1:
            return self._split_with_separators(text, separators[1:])

        candidate_chunks: list[str] = []
        current = ""
        for piece in pieces:
            next_piece = piece if not current else current + separator + piece
            if len(next_piece) <= self.config.chunk_size:
                current = next_piece
                continue
            if current:
                candidate_chunks.append(current)
            current = piece
        if current:
            candidate_chunks.append(current)

        results: list[str] = []
        for candidate in candidate_chunks:
            if len(candidate) <= self.config.chunk_size:
                results.append(candidate)
            else:
                results.extend(self._split_with_separators(candidate, separators[1:]))
        return self._apply_overlap(results)

    def _merge_small_pieces(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if len(current) + len(piece) <= self.config.chunk_size:
                current += piece
            else:
                if current:
                    chunks.append(current)
                current = piece
        if current:
            chunks.append(current)
        return self._apply_overlap(chunks)

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if not chunks or self.config.chunk_overlap <= 0:
            return [chunk.strip() for chunk in chunks if chunk.strip()]
        overlapped: list[str] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped.append(chunk.strip())
                continue
            prefix = chunks[idx - 1][-self.config.chunk_overlap :]
            overlapped.append((prefix + chunk).strip())
        return [chunk for chunk in overlapped if chunk]


class FixedSizeTextChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def split_text(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        step = max(1, self.config.chunk_size - self.config.chunk_overlap)
        while start < len(text):
            end = min(len(text), start + self.config.chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += step
        return chunks
