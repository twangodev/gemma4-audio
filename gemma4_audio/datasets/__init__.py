import os
from typing import Callable

# Enable parallel Rust-based downloads via hf_transfer before importing datasets
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from gemma4_audio.datasets.base import Dataset
from gemma4_audio.datasets.open_asr import OpenASRLeaderboardDataset

LIBRISPEECH_SPLITS = frozenset({"test.clean", "test.other"})

DATASET_REGISTRY: dict[str, Callable[[], Dataset]] = {
    "librispeech": lambda: OpenASRLeaderboardDataset(
        "librispeech", valid_splits=LIBRISPEECH_SPLITS
    ),
    "voxpopuli": lambda: OpenASRLeaderboardDataset("voxpopuli"),
    "ami": lambda: OpenASRLeaderboardDataset("ami"),
    "earnings22": lambda: OpenASRLeaderboardDataset("earnings22"),
    "gigaspeech": lambda: OpenASRLeaderboardDataset("gigaspeech"),
    "spgispeech": lambda: OpenASRLeaderboardDataset("spgispeech"),
    "tedlium": lambda: OpenASRLeaderboardDataset("tedlium"),
}


def get_dataset(name: str) -> Dataset:
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name]()


__all__ = ["DATASET_REGISTRY", "get_dataset"]
