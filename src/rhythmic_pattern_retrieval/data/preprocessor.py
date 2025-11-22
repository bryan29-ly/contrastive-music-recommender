import torch
import librosa
import certifi
from tqdm import tqdm
from demucs import pretrained
from demucs.apply import apply_model
import os

from rhythmic_pattern_retrieval.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_RATE, DURATION
from rhythmic_pattern_retrieval.utils.audio_utils import compute_mel_spectrogram, pad_or_crop_audio, resample_audio
from rhythmic_pattern_retrieval.utils.device import get_device

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()


class Preprocessor:
    def __init__(self, device=None, limit=None):
        self.device = device or get_device()
        self.limit = limit
        self.separator = pretrained.get_model("htdemucs")
        self.separator.to(self.device)

    def preprocess_file(self, mp3_path):
        save_path = PROCESSED_DATA_DIR / f"{mp3_path.stem}.pt"
        if save_path.exists():
            return None

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
        wav = wav.unsqueeze(0).to(self.device)

        ref_sr = self.separator.samplerate
        if sr != ref_sr:
            wav = resample_audio(wav, sr, ref_sr, self.device)

        # Separations Drums & Bass
        sources = apply_model(self.separator, wav, shifts=0)
        drums_audio = sources[0, 0, :, :]
        bass_audio = sources[0, 1, :, :]
        # Fusion
        rhythm_section = drums_audio + bass_audio
        # Mono
        rhythm_section = torch.mean(rhythm_section, dim=0)
        # Resample
        rhythm_section = resample_audio(
            rhythm_section, ref_sr, SAMPLE_RATE, self.device)
        # Pad
        rhythm_section = pad_or_crop_audio(
            rhythm_section, SAMPLE_RATE * DURATION)
        # Spectrogram
        spec_image = compute_mel_spectrogram(rhythm_section.cpu(), SAMPLE_RATE)

        # Save
        torch.save(spec_image, save_path)
        return save_path

    def preprocess_dataset(self):
        print(f"Device used : {self.device.type.upper()}.")
        audio_files = list(RAW_DATA_DIR.glob("**/*.mp3"))
        if self.limit:
            audio_files = audio_files[:self.limit]
        for mp3_path in tqdm(audio_files, desc="Processing"):
            try:
                self.preprocess_file(mp3_path)
            except Exception as e:
                print(f"Error on {mp3_path.name}: {e}")
        print(f"Done! Data in {PROCESSED_DATA_DIR}")
