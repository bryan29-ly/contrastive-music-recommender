import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import random


class ContrastiveAugmentations(nn.Module):
    def __init__(self, time_mask_param=40, freq_mask_param=30, noise_level=0.01, gain_range=0.2):
        super().__init__()
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)

        self.noise_level = noise_level
        self.gain_range = gain_range

    def forward(self, x):
        return self.augment(x)

    def augment(self, x):
        """
        x shape: [Batch, 1, Freq, Time]
        """
        out = x.clone()

        # 1. Random Gain (Volume)
        if self.gain_range > 0:
            gain_factor = random.uniform(
                1.0 - self.gain_range, 1.0 + self.gain_range)
            out = out * gain_factor

        # 2. Random Time Stretch
        if random.random() < 0.5:
            stretch_factor = random.uniform(0.85, 1.15)
            original_time_dim = out.shape[-1]
            new_time_dim = int(original_time_dim * stretch_factor)

            # Interpolation (Resize)
            out = F.interpolate(out, size=(
                out.shape[2], new_time_dim), mode='bilinear', align_corners=False)

            # Get back to the original size for batching
            if new_time_dim > original_time_dim:
                start = (new_time_dim - original_time_dim) // 2
                out = out[..., start:start+original_time_dim]
            else:
                pad_size = original_time_dim - new_time_dim
                out = F.pad(out, (0, pad_size), mode='constant', value=0)

        # 3. Masking
        out = self.time_masking(out)
        out = self.freq_masking(out)

        # 4. Gaussian Noise (Texture)
        if self.noise_level > 0:
            noise = torch.randn_like(out) * self.noise_level
            out = out + noise

        # Security (clamp)
        out = torch.clamp(out, min=0.0)

        return out
