import torch
import torch.nn as nn
import torchaudio.transforms as T
import random


class ContrastiveAugmentations(nn.Module):
    """Augmentation Pipeline. Take a spectrogram (Anchor) and results
      in two different versions (Views)
    """

    def __init__(self, time_mask_param=40, freq_mask_param=20, noise_level=0.01, gain_range=0.2):
        super().__init__()
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.noise_level = noise_level
        self.gain_range = gain_range

    def forward(self, x):
        """
        After take augment(crop1) & augment(crop2)
        """
        return self.augment(x)

    def augment(self, x):
        out = x.clone()
        # Random Gain
        if self.gain_range > 0:
            # Facteur entre (1.0 - range) et (1.0 + range)
            # Ex: entre 0.8 et 1.2
            gain_factor = random.uniform(
                1.0 - self.gain_range, 1.0 + self.gain_range)
            out = out * gain_factor

        # Masking
        out = self.time_masking(out)
        out = self.freq_masking(out)

        # Noise
        if self.noise_level > 0:
            noise = torch.randn_like(out) * self.noise_level
            out = out + noise

        # Security (clamp)
        out = torch.clamp(out, min=0.0)

        return out
