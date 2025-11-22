import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from demucs import pretrained
from demucs.apply import apply_model
import os
import certifi

from rhythmic_pattern_retrieval.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_RATE, DURATION

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_mel_spectrogram(audio_tensor, sr):
    if torch.is_tensor(audio_tensor):
        audio_np = audio_tensor.cpu().numpy()
    else:
        audio_np = audio_tensor

    if len(audio_np.shape) > 1:
        audio_np = np.mean(audio_np, axis=0)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_np, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / \
        (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return torch.tensor(mel_spec_db).float()


def preprocess_dataset(limit=None):
    device = get_device()
    print(f"Preprocessing started on {device.upper()}...")

    # Loading of Demucs
    print("Loading of Demucs...")
    separator = pretrained.get_model("htdemucs")
    separator.to(device)

    # Listing
    audio_files = list(RAW_DATA_DIR.glob("**/*.mp3"))
    if limit:
        audio_files = audio_files[:limit]

    print(f"Processing of {len(audio_files)} files...")

    for mp3_path in tqdm(audio_files, desc="Processing"):
        # try:
        save_path = PROCESSED_DATA_DIR / f"{mp3_path.stem}.pt"
        if save_path.exists():
            continue

        # Loading with Librosa
        wav_np, sr = librosa.load(str(mp3_path), sr=None, mono=False)

        # Convertion Numpy --> Tensor
        wav = torch.from_numpy(wav_np).float()

        # Librosa return (channels, time) for stereo, or (time,) for mono
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            # Become (2, time), artificial stereo for Demucs
            wav = wav.repeat(2, 1)
        elif wav.dim() == 2 and wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        # Add a dimension for Batch
        wav = wav.unsqueeze(0).to(device)

        ref_sr = separator.samplerate
        if sr != ref_sr:
            resampler = torchaudio.transforms.Resample(
                sr, ref_sr).to(device)
            wav = resampler(wav)

        # Separations
        sources = apply_model(separator, wav, shifts=0)

        # Drums extraction
        drums_audio = sources[0, 0, :, :]  # (Channels, Time)

        # Mixdown Mono
        drums_audio = torch.mean(drums_audio, dim=0)  # (Time,)

        # Resample final to 22050 Hz
        resampler_final = torchaudio.transforms.Resample(
            ref_sr, SAMPLE_RATE).to(device)
        drums_audio = resampler_final(drums_audio)

        # Pad
        max_samples = int(SAMPLE_RATE * DURATION)
        if drums_audio.shape[0] > max_samples:
            drums_audio = drums_audio[:max_samples]
        else:
            pad_size = max_samples - drums_audio.shape[0]
            drums_audio = torch.nn.functional.pad(
                drums_audio, (0, pad_size))

        # Spectrogram
        spec_image = compute_mel_spectrogram(
            drums_audio.cpu(), SAMPLE_RATE)
        print(f"save {mp3_path}")
        print(save_path)
        # Save
        torch.save(spec_image, save_path)

    # except Exception as e:
    #     print(f"\n❌ Error on {mp3_path.name}: {e}")
    #     continue
    # print(f"\n✅ Done ! Data in {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    # Test avec 2 fichiers pour voir si Librosa fait le job
    print(PROCESSED_DATA_DIR)
    preprocess_dataset(limit=8)
