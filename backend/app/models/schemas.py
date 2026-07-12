from typing import Any

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    chunk_method: str = Field(default="recursive")
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", "\u3002", "\uff0c", " ", ""])


class IngestResponse(BaseModel):
    document_ids: list[str]
    chunk_count: int
    filenames: list[str]


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    source_path: str
    parser_backend: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]


class DeleteDocumentResponse(BaseModel):
    document_id: str
    filename: str
    deleted_chunk_count: int


class IngestTaskCreateResponse(BaseModel):
    task_id: str


class IngestTaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    current_step: str
    elapsed_seconds: float
    filenames: list[str] = Field(default_factory=list)
    chunk_method: str = "recursive"
    detail: str = ""
    result: IngestResponse | None = None
    error: str | None = None


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    retrieval_sources: list[str]
    content: str
    metadata: dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    prompt: str


class EvalSample(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


class EvalRequest(BaseModel):
    samples: list[EvalSample]


class EvalResponse(BaseModel):
    metrics: dict[str, float]
