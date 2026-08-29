import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with an optional (possibly asymmetric) stride."""

    def __init__(self, in_ch, out_ch, stride=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positions.

    Sequence length differs between training (a fixed crop) and inference (a
    whole segment), so fixed positions are used rather than learned ones.
    """

    def __init__(self, dim, max_len=2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        inv_freq = torch.exp(torch.arange(0, dim, 2) *
                             (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * inv_freq)
        pe[:, 1::2] = torch.cos(position * inv_freq)
        # Fully determined by (dim, max_len), so keep it out of checkpoints.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:x.shape[1]]


class AttentivePooling(nn.Module):
    """Learned weighted average over time: [B, T, D] -> [B, D]."""

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1)
