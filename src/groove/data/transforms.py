import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class RhythmSafeAugment(nn.Module):
    """Contrastive augmentations on the log-mel [B, 1, F, T], applied on GPU.

    Every transform preserves the rhythmic structure: none reorders or removes
    beats. The model thus stays sensitive to groove while becoming invariant to
    tempo (mild), loudness, EQ/mastering, masking and spectral noise.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ts = cfg.time_stretch
        self.gain_range = cfg.gain_range
        self.eq = cfg.eq
        self.noise_std = cfg.noise_std
        self.n_time_masks = cfg.spec_masking.n_time_masks
        self.n_freq_masks = cfg.spec_masking.n_freq_masks
        # iid_masks=True -> an independent mask per item in the batch.
        self.time_mask = T.TimeMasking(
            cfg.spec_masking.time_mask_param, iid_masks=True)
        self.freq_mask = T.FrequencyMasking(
            cfg.spec_masking.freq_mask_param, iid_masks=True)

    def forward(self, x):
        b, _, n_mels, t = x.shape

        # 1. Time-stretch: mild tempo change, beat ordering preserved.
        #    One factor per call; the two views get separate calls -> different factors.
        if self.ts.enabled and random.random() < self.ts.prob:
            factor = random.uniform(self.ts.range[0], self.ts.range[1])
            new_t = max(1, int(round(t * factor)))
            x = F.interpolate(x, size=(n_mels, new_t),
                              mode="bilinear", align_corners=False)
            x = self._fit_time(x, t)

        # 2. Per-sample gain: loudness invariance (additive in log domain).
        if self.gain_range > 0:
            offset = (torch.rand(b, 1, 1, 1, device=x.device)
                      * 2 - 1) * self.gain_range
            x = x + offset

        # 3. Per-sample random EQ: mastering/timbre invariance, timing untouched.
        if self.eq.enabled:
            x = x + self._random_eq(b, n_mels, x.device)

        # 4. SpecAugment masking (light).
        for _ in range(self.n_time_masks):
            x = self.time_mask(x)
        for _ in range(self.n_freq_masks):
            x = self.freq_mask(x)

        # 5. Gaussian noise.
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        return x

    def _fit_time(self, x, target_t):
        """Crop or pad the time axis back to target_t after stretching."""
        t = x.shape[-1]
        if t > target_t:
            start = (t - target_t) // 2
            return x[..., start:start + target_t]
        if t < target_t:
            # Replicate rather than zero-pad: after standardization a zero block
            # is the dataset mean, a constant the model can key on.
            return F.pad(x, (0, target_t - t, 0, 0), mode="replicate")
        return x

    def _random_eq(self, b, n_mels, device):
        """A smooth random gain curve along the mel axis, one per sample."""
        ctrl = (torch.rand(b, 1, self.eq.n_bands, 1, device=device)
                * 2 - 1) * self.eq.strength
        # Interpolate the few control points into a full, smooth mel curve.
        curve = F.interpolate(ctrl, size=(n_mels, 1),
                              mode="bilinear", align_corners=True)
        return curve  # [B, 1, n_mels, 1] -> broadcasts over time
