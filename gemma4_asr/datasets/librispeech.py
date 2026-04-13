from typing import Iterator

import datasets as hf_datasets
import numpy as np

from gemma4_asr.audio import normalize_audio
from gemma4_asr.datasets.base import Sample


class LibriSpeechDataset:
    name: str = "librispeech"

    VALID_SPLITS = {"test-clean", "test-other", "dev-clean", "dev-other"}

    def __init__(self) -> None:
        self._data: hf_datasets.Dataset | None = None

    def load(self, split: str, seed: int = 42) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}' for LibriSpeech. "
                f"Valid splits: {sorted(self.VALID_SPLITS)}"
            )
        self._data = hf_datasets.load_dataset(
            "openslr/librispeech_asr",
            split=split.replace("-", "."),
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
                reference=row["text"].lower(),
            )

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("Call load() before getting length.")
        return len(self._data)
