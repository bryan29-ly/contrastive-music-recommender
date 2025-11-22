import torch
from pathlib import Path
from torch.utils.data import Dataset

from rhythmic_pattern_retrieval.config import PROCESSED_DATA_DIR


class SpectrogramDataset(Dataset):
    """Dataset for loading preprocessed drums+bass mel spectrograms."""

    def __init__(self, root_dir=PROCESSED_DATA_DIR, debug_limit=None):
        self.root_dir = Path(root_dir)

        print(f"Scanning spectrograms in {self.root_dir}...")
        self.spectrogram_files = list(self.root_dir.glob("*.pt"))

        if debug_limit is not None and len(self.spectrogram_files) > debug_limit:
            self.spectrogram_files = self.spectrogram_files[:debug_limit]

        if len(self.spectrogram_files) == 0:
            print(f"❌ WARNING: No .pt files found in {self.root_dir}.")
            print("   Run preprocessing first with Preprocessor class.")
        else:
            print(
                f"✅ Dataset initialized with {len(self.spectrogram_files)} spectrograms.")

    def __len__(self):
        return len(self.spectrogram_files)

    def __getitem__(self, index):
        """
        Returns:
            spectrogram (Tensor): Shape [1, 128, Time] (mel spectrogram)
            path (str): Path to the file
        """
        spec_path = self.spectrogram_files[index]

        try:
            # Load precomputed spectrogram
            spec = torch.load(spec_path)

            # Ensure shape is [1, 128, Time] for the model
            if spec.dim() == 2:
                spec = spec.unsqueeze(0)  # Add channel dimension

            return spec, str(spec_path)

        except Exception as e:
            print(f"❌ Error loading {spec_path}: {e}")
            # Return zeros to prevent crash (shape: [1, 128, 1000] as fallback)
            return torch.zeros((1, 128, 1000)), str(spec_path)


# class FMADataset(Dataset):
#     def __init__(self, root_dir=RAW_DATA_DIR, target_sample_rate=SAMPLE_RATE, duration=DURATION, debug_limit=None):
#         self.root_dir = Path(root_dir)
#         self.target_sample_rate = target_sample_rate
#         self.num_samples = target_sample_rate

#         print(f"Scanning files in {self.root_dir}...")
#         self.audio_files = list(self.root_dir.glob("**/*.mp3"))

#         # Security
#         if debug_limit is not None and len(self.audio_files) > debug_limit:
#             self.audio_files = self.audio_files[:debug_limit]

#         if len(self.audio_files) == 0:
#             print(f"❌ WARNING: No MP3 files found in {self.root_dir}.")
#         else:
#             print(
#                 f"✅ Dataset initialized with {len(self.audio_files)} tracks.")

#     def __len__(self):
#         return len(self.audio_files)

#     def __getitem__(self, index):
#         """
#         Returns:
#             waveform (Tensor): Shape [1, num_samples]
#             path (str): Path to the file (useful for debugging)
#         """
#         audio_path = self.audio_files[index]

#         try:
#             # 1. Load audio
#             # signal shape: (channels, time)
#             audio_np, _ = librosa.load(
#                 str(audio_path), sr=self.target_sample_rate, mono=True)
#             signal = torch.from_numpy(audio_np).float().unsqueeze(0)

#             # 2. Pad or Crop to fixed size (Critical for batch training)
#             if signal.shape[1] > self.num_samples:
#                 # Crop: Take the center part (often more representative)
#                 center = signal.shape[1] // 2
#                 start = center - (self.num_samples // 2)
#                 signal = signal[:, start: start + self.num_samples]

#             elif signal.shape[1] < self.num_samples:
#                 # Pad: Add zeros at the end
#                 padding = self.num_samples - signal.shape[1]
#                 signal = torch.nn.functional.pad(signal, (0, padding))

#             return signal, str(audio_path)

#         except Exception as e:
#             print(f"❌ Corrupted file {audio_path}: {e}")
#             # Return silent tensor to prevent crash
#             return torch.zeros((1, self.num_samples)), str(audio_path)
