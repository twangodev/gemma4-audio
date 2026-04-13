import librosa
import numpy as np

TARGET_SR = 16000


def normalize_audio(
    audio: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, int]:
    """Normalize audio to 16kHz mono float32 in [-1, 1]."""
    # Stereo to mono: average channels
    if audio.ndim == 2:
        audio = audio.mean(axis=0)

    # Resample if needed
    if sample_rate != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SR)

    # Ensure float32
    audio = audio.astype(np.float32)

    # Normalize to [-1, 1]
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio = audio / peak

    return audio, TARGET_SR
