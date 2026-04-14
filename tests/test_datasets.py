import pytest

from gemma4_audio.datasets import DATASET_REGISTRY, get_dataset
from gemma4_audio.datasets.open_asr import OpenASRLeaderboardDataset


OPEN_ASR_NAMES = [
    "voxpopuli",
    "ami",
    "earnings22",
    "gigaspeech",
    "spgispeech",
    "tedlium",
]


@pytest.mark.parametrize("name", OPEN_ASR_NAMES + ["librispeech"])
def test_registry_resolves(name: str):
    assert name in DATASET_REGISTRY
    ds = get_dataset(name)
    assert ds.name == name


def test_unknown_dataset_raises():
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_dataset("not-a-real-dataset")


@pytest.mark.parametrize("name", OPEN_ASR_NAMES + ["librispeech"])
def test_open_asr_rejects_bad_split(name: str):
    ds = get_dataset(name)
    with pytest.raises(ValueError, match="Invalid split"):
        ds.load("train")


def test_librispeech_accepts_dot_splits():
    ds = get_dataset("librispeech")
    for split in ("test.clean", "test.other"):
        assert split in ds._valid_splits
    assert "test-clean" not in ds._valid_splits


def test_open_asr_iter_without_load_raises():
    ds = OpenASRLeaderboardDataset("voxpopuli")
    with pytest.raises(RuntimeError, match="load"):
        next(iter(ds))
