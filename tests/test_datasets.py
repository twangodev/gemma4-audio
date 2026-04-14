import pytest

from gemma4_audio.datasets import DATASET_REGISTRY, get_dataset
from gemma4_audio.datasets.open_asr import OpenASRLeaderboardDataset


SHORTFORM_NAMES = [
    "voxpopuli",
    "ami",
    "earnings22",
    "gigaspeech",
    "spgispeech",
    "tedlium",
]

LONGFORM_NAMES = [
    "earnings21-long",
    "earnings22-long",
    "tedlium-long",
]

CORAAL_NAMES = [
    "coraal-atl",
    "coraal-dca",
    "coraal-dcb",
    "coraal-dta",
    "coraal-les",
    "coraal-prv",
    "coraal-roc",
    "coraal-vld",
]

ALL_OPEN_ASR_NAMES = SHORTFORM_NAMES + LONGFORM_NAMES + CORAAL_NAMES


@pytest.mark.parametrize("name", ALL_OPEN_ASR_NAMES + ["librispeech"])
def test_registry_resolves(name: str):
    assert name in DATASET_REGISTRY
    ds = get_dataset(name)
    assert ds.name == name


def test_unknown_dataset_raises():
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_dataset("not-a-real-dataset")


@pytest.mark.parametrize("name", ALL_OPEN_ASR_NAMES + ["librispeech"])
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


def test_coraal_uses_coraal_repo():
    ds = get_dataset("coraal-atl")
    assert ds._hf_repo == "bezzam/coraal"
    assert ds._config == "ATL"
    assert ds._id_field == "file_id"


def test_longform_uses_longform_repo():
    ds = get_dataset("earnings22-long")
    assert ds._hf_repo == "hf-audio/asr-leaderboard-longform"
    assert ds._config == "earnings22"
    assert ds._id_field is None
