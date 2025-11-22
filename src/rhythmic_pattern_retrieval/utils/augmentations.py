import torch
import torch.nn as nn
import torchaudio.transforms as T


class ContrastiveAugmentations(nn.Module):
    """Augmentation Pipeline. Take a spectrogram (Anchor) and results
      in two different versions (Views)
    """

    def __init__(self, time_mask_param=40, freq_mask_param=20, noise_level=0.01):
        super().__init__()
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.noise_level = noise_level

    def forward(self, x):
        """
        input : x (original spectrogram) [Batch, 1, 128, Time]
        output : (x_i, x_j) --> two augmented versions of the spectrogram
        """
        return self.augment(x), self.augment(x)

    def augment(self, x):
        out = x.clone()

        if torch.rand(1) < 0.5:
            shift = torch.randint(low=0, high=out.shape[-1], size=(1,)).item()
            out = torch.roll(out, shifts=shift, dims=-1)
        out = self.time_masking(out)
        out = self.freq_masking(out)
        if self.noise_level > 0:
            noise = torch.randn_like(out) * self.noise_level
            out = out + noise
        return out
