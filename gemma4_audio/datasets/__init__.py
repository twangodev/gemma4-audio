import os

# Enable parallel Rust-based downloads via hf_transfer before importing datasets
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from gemma4_audio.datasets.base import Dataset
from gemma4_audio.datasets.librispeech import LibriSpeechDataset

DATASET_REGISTRY: dict[str, type] = {
    "librispeech": LibriSpeechDataset,
}


def get_dataset(name: str) -> Dataset:
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {available}"
        )
    return DATASET_REGISTRY[name]()


__all__ = ["DATASET_REGISTRY", "get_dataset"]
