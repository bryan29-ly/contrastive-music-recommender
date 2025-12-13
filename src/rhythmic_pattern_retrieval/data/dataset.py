import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np

from rhythmic_pattern_retrieval.config import PROCESSED_DATA_DIR


class SpectrogramDataset(Dataset):
    """Dataset for loading preprocessed drums+bass mel spectrograms."""

    def __init__(self, root_dir=PROCESSED_DATA_DIR, crop_size=600, debug_limit=None):
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

    def _get_smart_crop(self, full_spec, valid_indices):
        time_steps = full_spec.shape[-1]

        # Padding if the file is too long
        if time_steps <= self.crop_size:
            pad_size = self.crop_size - time_steps
            return torch.nn.functional.pad(full_spec, (0, pad_size))

        # Starting choice
        if len(valid_indices) > 0:
            # Choose a timing
            anchor = np.random.choice(valid_indices)

            # Offset the anchor
            offset = random.randint(0, self.crop_size // 2)
            start = anchor - offset

            # Stay in the size of the file
            start = max(0, min(start, time_steps - self.crop_size))
        else:
            # Fallback
            start = random.randint(0, time_steps - self.crop_size)

        return full_spec[..., start: start + self.crop_size]

    def __getitem__(self, index):
        """
        Returns:
            spectrogram (Tensor): Shape [1, 128, Time] (mel spectrogram)
            path (str): Path to the file
        """
        spec_path = self.spectrogram_files[index]

        try:
            # Load precomputed spectrogram
            data = torch.load(spec_path, weights_only=False)
            if isinstance(data, dict):
                full_spec = data["mel"]
                valid_indices = data.get("valid_indices", [])
            else:
                full_spec = data
                valid_indices = []

            view1 = self._get_smart_crop(full_spec, valid_indices)
            view2 = self._get_smart_crop(full_spec, valid_indices)

            return view1, view2, str(spec_path)

        except Exception as e:
            print(f"❌ Error loading {spec_path}: {e}")
            # Return zeros to prevent crash (shape: [1, 128, 1000] as fallback)
            dummy = torch.zeros((1, 128, self.crop_size))
            return dummy, dummy, str(spec_path)
