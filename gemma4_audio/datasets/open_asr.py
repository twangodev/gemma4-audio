from typing import Iterator

import datasets as hf_datasets
import numpy as np

from gemma4_audio.audio import normalize_audio
from gemma4_audio.datasets.base import Sample


class OpenASRLeaderboardDataset:
    """Loader for ASR leaderboard test datasets on the HF Hub.

    Works across repos that follow the leaderboard's conventions — a uniform
    {audio, text, ...} schema per config/split. The defaults target
    hf-audio/open-asr-leaderboard; override hf_repo and id_field for the
    longform (hf-audio/asr-leaderboard-longform) and CORAAL (bezzam/coraal)
    repos.
    """

    DEFAULT_HF_REPO = "hf-audio/open-asr-leaderboard"

    def __init__(
        self,
        config: str,
        *,
        hf_repo: str = DEFAULT_HF_REPO,
        valid_splits: frozenset[str] = frozenset({"test"}),
        id_field: str | None = "id",
        display_name: str | None = None,
    ) -> None:
        self.name: str = display_name or config
        self._config: str = config
        self._hf_repo: str = hf_repo
        self._valid_splits: frozenset[str] = valid_splits
        self._id_field: str | None = id_field
        self._data: hf_datasets.Dataset | hf_datasets.IterableDataset | None = None

    def load(self, split: str, seed: int = 42, *, streaming: bool = False) -> None:
        if split not in self._valid_splits:
            raise ValueError(
                f"Invalid split '{split}' for {self.name}. "
                f"Valid splits: {sorted(self._valid_splits)}"
            )
        self._data = hf_datasets.load_dataset(
            self._hf_repo,
            self._config,
            split=split,
            streaming=streaming,
        ).shuffle(seed=seed)

    def __iter__(self) -> Iterator[Sample]:
        if self._data is None:
            raise RuntimeError("Call load() before iterating.")
        for i, row in enumerate(self._data):
            audio_array = np.array(row["audio"]["array"], dtype=np.float32)
            sr = row["audio"]["sampling_rate"]
            audio_norm, target_sr = normalize_audio(audio_array, sr)
            if self._id_field is not None and self._id_field in row:
                sample_id = str(row[self._id_field])
            else:
                sample_id = f"{self.name}-{i}"
            yield Sample(
                id=sample_id,
                audio=audio_norm,
                sample_rate=target_sr,
                reference=row["text"],
            )
