import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super.__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super.__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """x shape : [Batch, Time, 1]"""
        weights = self.attention(x)
        weights = F.softmax(weights, dim=1)
        # Context shape : [Batch, channels]
        context_vector = torch.sum(x * weights, dim=1)

        return context_vector, weights


class RhythmicEncoder(nn.Module):
    def __init__(self, projection_dim=128, base_channels=64):
        super.__init__()

        # 1. Stem (enter)
        # Kernel 3x3
        self.conv1 = nn.Conv2d(
            1, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.Relu()

        # Initial maxpool
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2. Backbone Resnet (4 layers)
        self.layer1 = self._make_layer(base_channels, base_channels, stride=1)
        self.layer2 = self._make_layer(
            base_channels, base_channels * 2, stride=2)
        self.layer3 = self._make_layer(
            base_channels * 2, base_channels * 4, stride=2)
        self.layer4 = self._make_layer(
            base_channels * 4, base_channels * 8, stride=2)

        # If base=64 shape layer 4 : 512 channels
        self.final_channels = base_channels * 8

        # 3. Attentio pooling
        self.attention_pool = AttentionPooling(self.final_channels)

        # 4. Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.final_channels, 512),
            nn.Relu(),
            nn.Linear(512, projection_dim)
        )

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResBlock(in_channels, out_channels, stride),
            ResBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        """x shape : [Batch, 1, freq, time]"""
        # Fetaure extraction (CNN)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Shape [Batch, 512, Fsmall, Tsmall]

        # Aggregation
        # 1. Meanon the remaining frequences
        x = x.mean(dim=2)  # [Batch, 512, Time]

        # 2. Permutation for attention
        # [Batch, Channels, Time] --> [Batch, Time, Channels]
        x = x.permute(0, 2, 1)

        # 3. Attention pooling on the time dimension
        h, attn_weights = self.attention_pool(x)  # h embeddings

        # Projection for NTXent loss
        z = self.projection_head(h)

        return h, z
