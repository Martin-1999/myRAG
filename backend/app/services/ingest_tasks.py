from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from backend.app.models.schemas import ChunkingConfig, IngestResponse
from backend.app.services.pipeline import RAGPipeline


@dataclass
class IngestTask:
    task_id: str
    status: str
    progress: int
    current_step: str
    started_at: float
    filenames: list[str]
    chunk_method: str
    detail: str = ""
    result: IngestResponse | None = None
    error: str | None = None


class IngestTaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, IngestTask] = {}
        self._lock = threading.Lock()

    def create_task(self, pipeline: RAGPipeline, files: list[Path], config: ChunkingConfig) -> str:
        task_id = uuid4().hex
        task = IngestTask(
            task_id=task_id,
            status="running",
            progress=0,
            current_step="queued",
            started_at=time.time(),
            filenames=[file.name for file in files],
            chunk_method=config.chunk_method,
        )
        with self._lock:
            self._tasks[task_id] = task

        worker = threading.Thread(
            target=self._run_task,
            args=(task_id, pipeline, files, config),
            daemon=True,
        )
        worker.start()
        return task_id

    def get_task(self, task_id: str) -> IngestTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def _update_task(self, task_id: str, **updates) -> None:
        with self._lock:
            task = self._tasks[task_id]
            for key, value in updates.items():
                setattr(task, key, value)

    def _run_task(self, task_id: str, pipeline: RAGPipeline, files: list[Path], config: ChunkingConfig) -> None:
        try:
            def progress_callback(progress: int, current_step: str, detail: str = "") -> None:
                self._update_task(task_id, progress=progress, current_step=current_step, detail=detail)

            document_ids, filenames, chunk_count = pipeline.ingest_files(files, config, progress_callback)
            result = IngestResponse(document_ids=document_ids, chunk_count=chunk_count, filenames=filenames)
            self._update_task(
                task_id,
                status="completed",
                progress=100,
                current_step="completed",
                detail="入库完成",
                result=result,
            )
        except Exception as exc:
            self._update_task(
                task_id,
                status="failed",
                current_step="failed",
                error=str(exc),
                detail="入库失败",
            )

    @staticmethod
    def elapsed_seconds(task: IngestTask) -> float:
        return round(time.time() - task.started_at, 2)
