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


# --- Temporal pooling heads. All take [B, C, T] and expose ``output_dim``. ---

class AttentionPooling(nn.Module):
    """Attention-weighted temporal pooling -> [B, C]."""

    def __init__(self, channels, hidden=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.output_dim = channels

    def forward(self, x):
        x = x.transpose(1, 2)                       # [B, T, C]
        weights = torch.softmax(self.attn(x), dim=1)  # [B, T, 1]
        return torch.sum(x * weights, dim=1)        # [B, C]


class StatsPooling(nn.Module):
    """Mean + std temporal pooling -> [B, 2C].

    The std captures rhythmic variability across the window, which a plain mean
    discards; useful to characterize groove.
    """

    def __init__(self, channels):
        super().__init__()
        self.output_dim = 2 * channels

    def forward(self, x):
        return torch.cat([x.mean(dim=-1), x.std(dim=-1)], dim=-1)


class GeMPooling(nn.Module):
    """Generalized-mean temporal pooling -> [B, C] (learnable exponent p)."""

    def __init__(self, channels, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps
        self.output_dim = channels

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=-1).pow(1.0 / self.p)


POOLINGS = {
    "attention": AttentionPooling,
    "stats": StatsPooling,
    "gem": GeMPooling,
}
