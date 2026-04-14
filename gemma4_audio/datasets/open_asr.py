from typing import Iterator

import datasets as hf_datasets
import numpy as np

from gemma4_audio.audio import normalize_audio
from gemma4_audio.datasets.base import Sample


class OpenASRLeaderboardDataset:
    """Loader for configs in the unified Open ASR Leaderboard test repo.

    Backed by hf-audio/open-asr-leaderboard — a single repo that mirrors the
    test splits of the ESB benchmark datasets with a uniform schema
    (audio, text, id, dataset, audio_length_s) and no gated-access friction.
    """

    HF_REPO = "hf-audio/open-asr-leaderboard"

    def __init__(self, config: str, valid_splits: frozenset[str] = frozenset({"test"})) -> None:
        self.name: str = config
        self._config: str = config
        self._valid_splits: frozenset[str] = valid_splits
        self._data: hf_datasets.Dataset | hf_datasets.IterableDataset | None = None

    def load(self, split: str, seed: int = 42, *, streaming: bool = False) -> None:
        if split not in self._valid_splits:
            raise ValueError(
                f"Invalid split '{split}' for {self.name}. "
                f"Valid splits: {sorted(self._valid_splits)}"
            )
        self._data = hf_datasets.load_dataset(
            self.HF_REPO,
            self._config,
            split=split,
            streaming=streaming,
        ).shuffle(seed=seed)

    def __iter__(self) -> Iterator[Sample]:
        if self._data is None:
            raise RuntimeError("Call load() before iterating.")
        for row in self._data:
            audio_array = np.array(row["audio"]["array"], dtype=np.float32)
            sr = row["audio"]["sampling_rate"]
            audio_norm, target_sr = normalize_audio(audio_array, sr)
            yield Sample(
                id=str(row["id"]),
                audio=audio_norm,
                sample_rate=target_sr,
                reference=row["text"],
            )