from pathlib import Path

from modelscope import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_IDS = [
    "OpenDataLab/MinerU2.5-Pro-2605-1.2B",
    "Qwen/Qwen3-Embedding-0.6B",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
]


def download_model(model_id: str) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return snapshot_download(model_id, local_dir=str(MODELS_DIR))


if __name__ == "__main__":
    for model_id in MODEL_IDS:
        model_dir = download_model(model_id)
        print(f"Downloaded {model_id} to: {model_dir}")
