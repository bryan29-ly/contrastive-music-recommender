import torch
import torchaudio
import certifi
import os
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from demucs import pretrained
from demucs.apply import apply_model

from rhythmic_pattern_retrieval.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_RATE, DURATION
from rhythmic_pattern_retrieval.utils.audio_utils import compute_mel_spectrogram, pad_or_crop_audio, resample_audio
from rhythmic_pattern_retrieval.utils.device import get_device

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()


# Performance parameters
BATCH_SIZE = 32       # Increase if you have a large GPU
NUM_WORKERS = 16      # Use all available CPU cores
DEMUCS_SR = 44100     # Demucs target sample rate
TARGET_LEN = int(SAMPLE_RATE * DURATION) # Final spectrogram length

class MP3Dataset(Dataset):
    """
    Fast parallel loading using Torchaudio (C++ backend).
    """
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mp3_path = self.file_list[idx]
        try:
            # Load audio with Torchaudio (faster than Librosa)
            wav, sr = torchaudio.load(str(mp3_path))

            # Ensure 2 channels for Demucs
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2, :] # On garde juste L/R si multicanal

            # Resample to 44100 Hz if needed
            if sr != DEMUCS_SR:
                wav = torchaudio.functional.resample(wav, sr, DEMUCS_SR)

            # Pad/Crop for consistent tensor size
            target_samples = int(DEMUCS_SR * DURATION) # ~30s * 44100
            
            if wav.shape[-1] > target_samples:
                wav = wav[..., :target_samples]
            else:
                pad_size = target_samples - wav.shape[-1]
                wav = torch.nn.functional.pad(wav, (0, pad_size))
                
            return wav, str(mp3_path)
            
        except Exception as e:
            # On error, return empty tensor (filtered later)
            return torch.zeros(2, int(DEMUCS_SR * DURATION)), ""

class BatchPreprocessor:
    def __init__(self, limit=None):
        self.device = get_device()
        print(f"Initializing Demucs on {self.device}...")
        
    # Load Demucs model
        self.separator = pretrained.get_model("htdemucs")
        self.separator.to(self.device)
        self.separator.eval()
        
    # List files
        self.all_files = list(RAW_DATA_DIR.glob("**/*.mp3"))
        if limit:
            self.all_files = self.all_files[:limit]
            
        print(f"Found {len(self.all_files)} files to process.")

    def process_batch(self, batch_wavs, batch_paths):
        """
        Traite un lot de 32 musiques d'un coup sur le GPU.
        Tout reste sur le GPU jusqu'à la sauvegarde.
        """
        
    # Move batch to GPU
        batch_wavs = batch_wavs.to(self.device)
        
        # Demucs batch inference
        with torch.no_grad():
            sources = apply_model(self.separator, batch_wavs, shifts=0)
        
    # Extract drums and bass, mix stereo to mono
        drums = sources[:, 0, :, :].mean(dim=1) # [Batch, Time]
        bass = sources[:, 1, :, :].mean(dim=1)

    # Resample batch to 22kHz
        drums = resample_audio(drums, DEMUCS_SR, SAMPLE_RATE, self.device)
        bass = resample_audio(bass, DEMUCS_SR, SAMPLE_RATE, self.device)

    # Crop/Pad batch for safety
        if drums.shape[-1] != TARGET_LEN:
             drums = pad_or_crop_audio(drums, TARGET_LEN)
             bass = pad_or_crop_audio(bass, TARGET_LEN)

    # Compute mel spectrograms
        drums_spec = compute_mel_spectrogram(drums, SAMPLE_RATE)
        bass_spec = compute_mel_spectrogram(bass, SAMPLE_RATE)

    # Stack drums and bass: [Batch, 2, Freq, Time]
        final_batch = torch.stack([drums_spec, bass_spec], dim=1)

    # Save tensors to disk
        for i, path_str in enumerate(batch_paths):
            if path_str == "": continue # Skip errors
            
            p = Path(path_str)
            save_path = PROCESSED_DATA_DIR / f"{p.stem}.pt"
            
            # Move tensor to CPU before saving
            torch.save(final_batch[i].cpu(), save_path)

    def run(self):
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset & DataLoader
        dataset = MP3Dataset(self.all_files)
        
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        print(f"Starting Batch Processing (Batch: {BATCH_SIZE}, Workers: {NUM_WORKERS})")
        
    # Main loop
        for batch_wavs, batch_paths in tqdm(loader, desc="Processing Batches"):
            # Skip batch if all files are invalid
            valid_mask = [p != "" for p in batch_paths]
            if not any(valid_mask): continue
            
            self.process_batch(batch_wavs, batch_paths)

if __name__ == "__main__":
    # Désactive le gradient globalement (Gain VRAM énorme)
    torch.set_grad_enabled(False)
    
    processor = BatchPreprocessor(limit=None)
    processor.run()