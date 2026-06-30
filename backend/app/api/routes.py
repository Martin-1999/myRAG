from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.models.schemas import (
    AskRequest,
    AskResponse,
    ChunkingConfig,
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentRecord,
    EvalRequest,
    EvalResponse,
    IngestTaskCreateResponse,
    IngestTaskStatusResponse,
    IngestResponse,
    RetrievedChunk,
)
from backend.app.services.evaluator import RagasEvaluator
from backend.app.services.ingest_tasks import IngestTaskManager
from backend.app.services.pipeline import RAGPipeline


router = APIRouter()
task_manager = IngestTaskManager()


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@lru_cache
def get_evaluator() -> RagasEvaluator:
    return RagasEvaluator()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: list[UploadFile] = File(...),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(100),
) -> IngestResponse:
    pipeline = get_pipeline()
    upload_paths: list[Path] = []
    upload_root = pipeline.settings.upload_dir
    for file in files:
        destination = upload_root / file.filename
        content = await file.read()
        destination.write_bytes(content)
        upload_paths.append(destination)
    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document_ids, filenames, chunk_count = pipeline.ingest_files(upload_paths, config)
    return IngestResponse(document_ids=document_ids, chunk_count=chunk_count, filenames=filenames)


@router.post("/ingest/tasks", response_model=IngestTaskCreateResponse)
async def create_ingest_task(
    files: list[UploadFile] = File(...),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(100),
) -> IngestTaskCreateResponse:
    pipeline = get_pipeline()
    upload_paths: list[Path] = []
    upload_root = pipeline.settings.upload_dir
    for file in files:
        destination = upload_root / file.filename
        content = await file.read()
        destination.write_bytes(content)
        upload_paths.append(destination)
    config = ChunkingConfig(
        chunk_method=chunk_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    task_id = task_manager.create_task(pipeline, upload_paths, config)
    return IngestTaskCreateResponse(task_id=task_id)


@router.get("/ingest/tasks/{task_id}", response_model=IngestTaskStatusResponse)
async def get_ingest_task(task_id: str) -> IngestTaskStatusResponse:
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} was not found.")
    return IngestTaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        current_step=task.current_step,
        elapsed_seconds=task_manager.elapsed_seconds(task),
        filenames=task.filenames,
        chunk_method=task.chunk_method,
        detail=task.detail,
        result=task.result,
        error=task.error,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    pipeline = get_pipeline()
    documents = [DocumentRecord(**item) for item in pipeline.list_documents()]
    return DocumentListResponse(documents=documents)


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(document_id: str) -> DeleteDocumentResponse:
    pipeline = get_pipeline()
    try:
        deleted = pipeline.delete_document(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DeleteDocumentResponse(**deleted)


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    pipeline = get_pipeline()
    answer, retrieved, prompt = await pipeline.ask(request.question, request.top_k)
    return AskResponse(
        answer=answer,
        retrieved_chunks=[RetrievedChunk(**item) for item in retrieved],
        prompt=prompt,
    )


@router.post("/evaluate", response_model=EvalResponse)
async def evaluate_rag(request: EvalRequest) -> EvalResponse:
    evaluator = get_evaluator()
    samples = [
        {
            "question": sample.question,
            "answer": sample.answer,
            "contexts": sample.contexts,
            "ground_truth": sample.ground_truth,
        }
        for sample in request.samples
    ]
    metrics = evaluator.evaluate(samples)
    return EvalResponse(metrics=metrics)
