import os
from typing import Callable

# Enable parallel Rust-based downloads via hf_transfer before importing datasets
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from gemma4_audio.datasets.base import Dataset
from gemma4_audio.datasets.open_asr import OpenASRLeaderboardDataset

LIBRISPEECH_SPLITS = frozenset({"test.clean", "test.other"})
LONGFORM_REPO = "hf-audio/asr-leaderboard-longform"
CORAAL_REPO = "bezzam/coraal"
CORAAL_SUBSETS = ("ATL", "DCA", "DCB", "DTA", "LES", "PRV", "ROC", "VLD")

DATASET_REGISTRY: dict[str, Callable[[], Dataset]] = {
    # Short-form (hf-audio/open-asr-leaderboard)
    "librispeech": lambda: OpenASRLeaderboardDataset(
        "librispeech", valid_splits=LIBRISPEECH_SPLITS
    ),
    "voxpopuli": lambda: OpenASRLeaderboardDataset("voxpopuli"),
    "ami": lambda: OpenASRLeaderboardDataset("ami"),
    "earnings22": lambda: OpenASRLeaderboardDataset("earnings22"),
    "gigaspeech": lambda: OpenASRLeaderboardDataset("gigaspeech"),
    "spgispeech": lambda: OpenASRLeaderboardDataset("spgispeech"),
    "tedlium": lambda: OpenASRLeaderboardDataset("tedlium"),
    # Long-form (hf-audio/asr-leaderboard-longform) — samples are 30–60 min each
    "earnings21-long": lambda: OpenASRLeaderboardDataset(
        "earnings21",
        hf_repo=LONGFORM_REPO,
        id_field=None,
        display_name="earnings21-long",
    ),
    "earnings22-long": lambda: OpenASRLeaderboardDataset(
        "earnings22",
        hf_repo=LONGFORM_REPO,
        id_field=None,
        display_name="earnings22-long",
    ),
    "tedlium-long": lambda: OpenASRLeaderboardDataset(
        "tedlium",
        hf_repo=LONGFORM_REPO,
        id_field=None,
        display_name="tedlium-long",
    ),
}

# CORAAL dialect subsets (bezzam/coraal), e.g. "coraal-atl" → config "ATL"
for _sub in CORAAL_SUBSETS:
    _name = f"coraal-{_sub.lower()}"
    DATASET_REGISTRY[_name] = (
        lambda sub=_sub, name=_name: OpenASRLeaderboardDataset(
            sub,
            hf_repo=CORAAL_REPO,
            id_field="file_id",
            display_name=name,
        )
    )


def get_dataset(name: str) -> Dataset:
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name]()


__all__ = ["DATASET_REGISTRY", "get_dataset"]