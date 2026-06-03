import librosa
import torch


def get_duration(path):
    """Return the duration of an audio file in seconds (no full decode)."""
    return librosa.get_duration(path=str(path))


def load_audio(path, target_sr, mono=True):
    """Decode an audio file to a 1-D float32 tensor at ``target_sr``.

    Used offline only (not perf-critical), so we rely on librosa which decodes
    mp3/wav/flac robustly across platforms and resamples in one call.
    """
    wav, _ = librosa.load(str(path), sr=target_sr, mono=mono)
    return torch.from_numpy(wav).float()
