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

    def _pad_if_needed(self, spec):
        if spec.shape[-1] < self.crop_size:
            pad_size = self.crop_size - spec.shape[-1]
            return torch.nn.functional.pad(spec, (0, pad_size))
        return spec

    def __getitem__(self, index):
        """
        Returns:
            spectrogram (Tensor): Shape [1, 128, Time] (mel spectrogram)
            path (str): Path to the file
        """
        spec_path = self.spectrogram_files[index]

        try:
            # 1. Load precomputed spectrogram
            data = torch.load(spec_path, weights_only=False)
            if isinstance(data, dict):
                full_spec = data["mel"]
                valid_indices = data.get("valid_indices", [])
            else:
                full_spec = data
                valid_indices = []

            time_steps = full_spec.shape[-1]

            # 2. Base anchor point
            # if file is too short
            if time_steps <= self.crop_size:
                base_start = 0
            # RMS valid zones
            elif len(valid_indices) > 0:
                anchor = np.random.choice(valid_indices)
                offset = random.randint(0, self.crop_size // 2)
                base_start = anchor - offset
            # If no RMS info
            else:
                base_start = random.randint(
                    0, max(0, time_steps - self.crop_size))

            # Security on base_start
            max_start = max(0, time_steps - self.crop_size)
            base_start = max(0, min(base_start, max_start))

            # 3. Views with Jitter (temporale shift)
            jitter_range = 25  # +/- 25 frames

            # View 1 : base + shift
            shift1 = random.randint(-jitter_range, jitter_range)
            start1 = max(0, min(base_start + shift1,
                         time_steps - self.crop_size))
            view1_crop = full_spec[..., start1: start1 + self.crop_size]

            # View 2 : base + shift
            shift2 = random.randint(-jitter_range, jitter_range)
            start2 = max(0, min(base_start + shift2,
                         time_steps - self.crop_size))
            view2_crop = full_spec[..., start2: start2 + self.crop_size]

            # 4. Final padding
            view1 = self._pad_if_needed(view1_crop)
            view2 = self._pad_if_needed(view2_crop)

            return view1, view2, str(spec_path)

        except Exception as e:
            print(f"❌ Error loading {spec_path}: {e}")
            # Return zeros to prevent crash (shape: [1, 128, 1000] as fallback)
            dummy = torch.zeros((1, 128, self.crop_size))
            return dummy, dummy, str(spec_path)
