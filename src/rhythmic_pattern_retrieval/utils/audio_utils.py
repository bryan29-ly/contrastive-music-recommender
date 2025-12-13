import torch
import torchaudio
import librosa
import numpy as np


def pad_or_crop_audio(audio, max_samples):
    if audio.shape[0] > max_samples:
        return audio[:max_samples]
    else:
        pad_size = max_samples - audio.shape[0]
        return torch.nn.functional.pad(audio, (0, pad_size))


def resample_audio(audio, orig_sr, target_sr, device):
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr).to(device)
    return resampler(audio)


def compute_mel_spectrogram(audio_tensor, sr, n_mels=128, n_fft=2048, hop_length=512):
    if torch.is_tensor(audio_tensor):
        audio_np = audio_tensor.cpu().numpy()
    else:
        audio_np = audio_tensor

    if len(audio_np.shape) > 1:
        audio_np = np.mean(audio_np, axis=0)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_np, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / \
        (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return torch.tensor(mel_spec_db).float()


def get_valid_frames(audio_np, hop_length=512, threshold_ratio=0.4):
    if len(audio_np.shape) > 1:
        audio_np = np.mean(audio_np, axis=0)
    # Compute RMS
    rms = librosa.feature.rms(
        y=audio_np, frame_length=2048, hop_length=hop_length)[0]
    # Set a threshold
    threshold = threshold_ratio * np.max(rms)
    # Get the meaningful frames
    valid_indices = np.where(rms > threshold)

    return valid_indices
