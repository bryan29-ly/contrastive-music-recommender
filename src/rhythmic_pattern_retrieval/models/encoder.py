import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. RESNET BLOCK


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super.__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # if we change the size (stride) or the numbers of filters
        self.shortcut = nn.Sequential()

        if stride != (1, 1) or in_channels != out_channels:
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


# 2. Attention Pooling
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super.__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """x shape : [Batch, Time, Channels]"""
        weights = self.attention(x)
        weights = F.softmax(weights, dim=1)
        # Context shape : [Batch, channels]
        context_vector = torch.sum(x * weights, dim=1)

        return context_vector, weights


# 3. Principal encoder
class RhythmicEncoder(nn.Module):
    def __init__(self, projection_dim=128):
        super.__init__()

        # 1. Stem (enter)
        # Kernel 5x5, stride (2, 1)
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=(5, 5), stride=(2, 1), padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Initial maxpool
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)

        # 2. Backbone Resnet (4 layers), layers 3 and 4 stride to reduce only the frequencies
        self.layer1 = self._make_layer(64, 64, stride=(1, 1))
        self.layer2 = self._make_layer(64, 128, stride=(2, 2))
        self.layer3 = self._make_layer(128, 256, stride=(2, 1))
        self.layer4 = self._make_layer(256, 512, stride=(2, 1))

        # 3. Attention pooling
        self.attention_pool = AttentionPooling(512)

        # 4. Projection head for simCLR
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def _make_layer(self, in_channels, out_channels, stride):
        # Two resnet blocks, reduce then stabilize
        return nn.Sequential(
            ResBlock(in_channels, out_channels, stride=stride),
            ResBlock(out_channels, out_channels, stride=(1, 1))
        )

    def forward(self, x):
        """x shape : [Batch, 1, freq, time]"""
        # Fetaure extraction (CNN)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x)))
                         )  # [B, 64, 32, 300]

        x = self.layer1(x)
        x = self.layer2(x)  # [B, 128, 16, 150]
        x = self.layer3(x)  # [B, 256, 8, 150]
        x = self.layer4(x)  # [B, 512, 4, 150]

        # Aggregation
        # 1. Meanon the remaining frequences
        x = x.mean(dim=2)  # [Batch, 512, 150]

        # 2. Permutation for attention
        x = x.permute(0, 2, 1)  # [B, 150, 512]

        # 3. Attention pooling on the time dimension
        h, weights = self.attention_pool(x)  # h embeddings

        # Projection for NTXent loss only
        z = self.projection_head(h)

        return h, z
