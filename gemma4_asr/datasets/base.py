from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Sample:
    id: str
    audio: np.ndarray
    sample_rate: int
    reference: str


@runtime_checkable
class Dataset(Protocol):
    name: str

    def load(self, split: str, seed: int = 42) -> None: ...
    def __iter__(self) -> Iterator[Sample]: ...
    def __len__(self) -> int: ...
