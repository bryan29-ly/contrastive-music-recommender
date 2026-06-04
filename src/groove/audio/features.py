from contextlib import nullcontext

import torch
import torch.nn as nn
import torchaudio.transforms as T


class LogMelSpectrogram(nn.Module):
    """Log-mel spectrogram, instantiated once and reused (GPU-friendly).

    Input : waveform [B, samples] (or [samples])
    Output: mel [B, 1, n_mels, time]

    The transform has no learnable parameters, so it runs under ``no_grad``:
    we never backprop into the raw audio, only from the encoder onwards.
    """

    def __init__(self, cfg):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=cfg.power,
            norm=cfg.mel_norm,
            mel_scale=cfg.mel_scale,
            center=True,
            pad_mode="reflect",
        )
        # Fixed log-compression. We deliberately avoid AmplitudeToDB, whose
        # top_db floor is relative to the per-tensor max and would make the
        # output depend on batch composition (train/inference inconsistency).
        self.log_eps = 1e-5
        # Global standardization stats (computed once over the dataset).
        # None until estimated -> the normalization step is skipped.
        self.norm_mean = cfg.norm_mean
        self.norm_std = cfg.norm_std

    @torch.no_grad()
    def forward(self, wav):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # Compute the STFT in fp32 even under AMP: cuFFT is unstable in fp16.
        # Only needed to override an enclosing CUDA autocast; elsewhere (MPS/CPU)
        # a plain context avoids constructing an unsupported autocast.
        amp_off = (torch.autocast(device_type="cuda", enabled=False)
                   if wav.device.type == "cuda" else nullcontext())
        with amp_off:
            mel = self.mel(wav.float())
            mel = torch.log(mel + self.log_eps)  # batch-independent compression

        if self.norm_mean is not None and self.norm_std is not None:
            mel = (mel - self.norm_mean) / (self.norm_std + 1e-8)

        return mel.unsqueeze(1)  # [B, 1, n_mels, time]
