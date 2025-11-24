import torch
import random
from pathlib import Path
from torch.utils.data import Dataset

from rhythmic_pattern_retrieval.config import PROCESSED_DATA_DIR


class SpectrogramDataset(Dataset):
    """Dataset for loading preprocessed drums+bass mel spectrograms."""

    def __init__(self, root_dir=PROCESSED_DATA_DIR, crop_size=256, debug_limit=None):
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size

        print(f"Scanning spectrograms in {self.root_dir}...")
        self.spectrogram_files = list(self.root_dir.glob("*.pt"))

        if debug_limit is not None and len(self.spectrogram_files) > debug_limit:
            self.spectrogram_files = self.spectrogram_files[:debug_limit]

        if len(self.spectrogram_files) == 0:
            print(f"❌ WARNING: No .pt files found in {self.root_dir}.")
        else:
            print(
                f"✅ Dataset initialized with {len(self.spectrogram_files)} spectrograms.")

    def __len__(self):
        return len(self.spectrogram_files)

    def _random_crop(self, spec):
        _, _, time_steps = spec.shape
        if self.crop_size is None or time_steps <= self.crop_size:
            return spec[:, :, :self.crop_size]
        else:
            start = random.randint(0, time_steps - self.crop_size)
            return spec[:, :, start: start + self.crop_size]

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
            spec = self._random_crop(spec)
            return spec, str(spec_path)

        except Exception as e:
            print(f"❌ Error loading {spec_path}: {e}")
            # Return zeros to prevent crash (shape: [1, 128, 1000] as fallback)
            return torch.zeros((2, 128, 1000)), str(spec_path)
