from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from PIL import Image


@dataclass
class ParsedDocument:
    document_id: str
    filename: str
    text: str
    tables: list[str]
    metadata: dict


class DocumentParser:
    def __init__(
        self,
        parsed_dir: Path,
        backend: str = "mineru",
        model_path: Path | None = None,
        device_map: str = "auto",
        image_analysis: bool = False,
        pdf_dpi: int = 144,
    ) -> None:
        self.parsed_dir = parsed_dir
        self.backend = backend
        self.model_path = model_path
        self.device_map = device_map
        self.image_analysis = image_analysis
        self.pdf_dpi = pdf_dpi

    def parse_pdf(self, file_path: Path) -> ParsedDocument:
        if self.backend.lower() != "mineru":
            raise RuntimeError("This project is configured for local MinerU parsing only.")
        return self._parse_with_local_model(file_path)

    def _parse_with_local_model(self, file_path: Path, progress_callback=None) -> ParsedDocument:
        model_path = self._resolve_model_path()
        client = get_mineru_client(str(model_path), self.device_map, self.image_analysis)

        page_texts: list[str] = []
        tables: list[str] = []

        with TemporaryDirectory(prefix="mineru-pages-") as temp_dir:
            image_paths = self._render_pdf_pages(file_path, Path(temp_dir))
            total_pages = max(1, len(image_paths))
            for page_number, image_path in enumerate(image_paths, start=1):
                if progress_callback:
                    progress_callback(page_number, total_pages)
                with Image.open(image_path) as page_image:
                    result = client.two_step_extract(page_image)
                normalized = self._normalize_result(result)
                page_text = normalized["text"].strip()
                if page_text:
                    page_texts.append(f"[Page {page_number}]\n{page_text}")
                tables.extend(normalized["tables"])

        document_id = uuid4().hex
        payload = ParsedDocument(
            document_id=document_id,
            filename=file_path.name,
            text="\n\n".join(page_texts).strip(),
            tables=tables,
            metadata={
                "source_path": str(file_path),
                "parser_backend": "mineru_local_python",
                "mineru_model_path": str(model_path),
                "image_analysis": self.image_analysis,
            },
        )
        self._persist(payload)
        return payload

    def _resolve_model_path(self) -> Path:
        if self.model_path and self.model_path.exists():
            return self.model_path.resolve()
        raise RuntimeError(
            "Local MinerU model path does not exist. Set MINERU_MODEL_PATH to your downloaded "
            "MinerU2.5-Pro model directory."
        )

    def _render_pdf_pages(self, file_path: Path, output_dir: Path) -> list[Path]:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is required to convert uploaded PDFs into page images. "
                "Install `pymupdf` in the current environment."
            ) from exc

        pdf = fitz.open(file_path)
        scale = self.pdf_dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        image_paths: list[Path] = []
        try:
            for page_index in range(pdf.page_count):
                page = pdf.load_page(page_index)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image_path = output_dir / f"page_{page_index + 1}.png"
                pix.save(image_path)
                image_paths.append(image_path)
        finally:
            pdf.close()
        return image_paths

    def _normalize_result(self, result: object) -> dict[str, list[str] | str]:
        if isinstance(result, str):
            return {"text": result, "tables": []}

        if isinstance(result, dict):
            text_parts: list[str] = []
            table_parts: list[str] = []
            self._collect_result_parts(result, text_parts, table_parts)
            return {"text": "\n\n".join(text_parts), "tables": table_parts}

        if isinstance(result, list):
            text_parts: list[str] = []
            table_parts: list[str] = []
            for item in result:
                if isinstance(item, dict):
                    self._collect_result_parts(item, text_parts, table_parts)
                elif item:
                    text_parts.append(str(item).strip())
            return {"text": "\n\n".join(text_parts), "tables": table_parts}

        return {"text": str(result), "tables": []}

    def _collect_result_parts(self, payload: dict, text_parts: list[str], table_parts: list[str]) -> None:
        for key in ("text", "content", "markdown", "md", "html", "latex"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())

        for key in ("tables", "table_res_list"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        merged = "\n".join(
                            str(item.get(field, "")).strip()
                            for field in ("text", "content", "html", "latex", "table_caption", "table_body")
                            if str(item.get(field, "")).strip()
                        )
                        if merged:
                            table_parts.append(merged)
                    elif item:
                        table_parts.append(str(item).strip())

        for key in ("pages", "results", "blocks", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._collect_result_parts(item, text_parts, table_parts)

    def _persist(self, parsed_document: ParsedDocument) -> None:
        output_path = self.parsed_dir / f"{parsed_document.document_id}.json"
        output_path.write_text(
            json.dumps(
                {
                    "document_id": parsed_document.document_id,
                    "filename": parsed_document.filename,
                    "text": parsed_document.text,
                    "tables": parsed_document.tables,
                    "metadata": parsed_document.metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


@lru_cache
def get_mineru_client(model_path: str, device_map: str, image_analysis: bool):
    from mineru_vl_utils import MinerUClient
    from modelscope import AutoProcessor, Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    return MinerUClient(
        backend="transformers",
        model=model,
        processor=processor,
        image_analysis=image_analysis,
    )
