import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import numpy as np


def pad_or_crop_audio(audio, max_samples):
    if audio.shape[0] > max_samples:
        return audio[:max_samples]
    else:
        pad_size = max_samples - audio.shape[0]
        return torch.nn.functional.pad(audio, (0, pad_size))


def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio

    if audio.device.type != 'cpu':
        audio = audio.cpu()

    return F.resample(audio, orig_sr, target_sr)


def compute_mel_spectrogram(audio_tensor, sr, n_mels=128, n_fft=2048, hop_length=256):
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk"
    )

    to_db_transform = T.AmplitudeToDB(stype="power", top_db=80)

    if audio_tensor.device.type != 'cpu':
        mel_spectrogram_transform = mel_spectrogram_transform.to(
            audio_tensor.device)
        to_db_transform = to_db_transform.to(audio_tensor.device)

    mel_spec = mel_spectrogram_transform(audio_tensor)
    mel_spec_db = to_db_transform(mel_spec)

    min_val = mel_spec_db.min()
    max_val = mel_spec_db.max()
    mel_spec_db = (mel_spec_db - min_val) / (max_val - min_val + 1e-8)

    return mel_spec_db


def get_valid_frames(audio_np, hop_length=256, threshold_ratio=0.4):
    if len(audio_np.shape) > 1:
        audio_np = np.mean(audio_np, axis=0)
    # Compute RMS
    rms = librosa.feature.rms(
        y=audio_np, frame_length=2048, hop_length=hop_length)[0]
    # Set a threshold
    threshold = threshold_ratio * np.max(rms)
    # Get the meaningful frames
    valid_indices = np.where(rms > threshold)[0]

    return valid_indices
