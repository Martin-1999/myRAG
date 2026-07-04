from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "myRAG"
    api_prefix: str = "/api"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://127.0.0.1:5173"])

    upload_dir: Path = BASE_DIR / "data" / "uploads"
    parsed_dir: Path = BASE_DIR / "data" / "parsed"
    chroma_dir: Path = BASE_DIR / "data" / "chroma"
    chunk_store_path: Path = BASE_DIR / "data" / "parsed" / "chunks.jsonl"

    mineru_backend: str = "mineru"
    mineru_model_path: Path = BASE_DIR / "models" / "OpenDataLab" / "MinerU2___5-Pro-2605-1___2B"
    mineru_device_map: str = "auto"
    mineru_image_analysis: bool = False
    mineru_pdf_dpi: int = 144

    embedding_model_name: str = "Qwen/Qwen3-Embedding-8B"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8
    dense_top_k: int = 8
    sparse_top_k: int = 8
    bm25_top_k: int = 8
    final_top_k: int = 5
    rrf_k: int = 60

    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"

    llm_api_base: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model_name: str = "deepseek-chat"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    llm_timeout_seconds: int = 120

    chroma_collection_name: str = "myRAG_documents"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return settings
