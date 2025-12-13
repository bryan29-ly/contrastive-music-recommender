import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        weights = self.attention(x)
        weights = F.softmax(weights, dim=1)

        context_vector = torch.sum(x * weights, dim=1)
        return context_vector, weights


class RhythmicEncoder(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()

        # Load Resnet18 with no weights
        resnet = resnet18(weights=None)

        # Modify the first layer to accept 1 channel (Spectrogram)
        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=(2, 1),
            padding=3,
            bias=False
        )

        # MaxPool
        resnet.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)

        # Temporal stride to 1
        resnet.layer2[0].conv1.stride = (2, 1)
        resnet.layer2[0].downsample[0].stride = (2, 1)

        self.cnn_backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )

        # Output dimension
        cnn_out_channels = 128

        # RNN GRU
        self.rnn = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Attention
        self.attention = AttentionBlock(hidden_size=256 * 2)

        # Projection for Contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        features = self.cnn_backbone(x)

        # Frequency pooling
        features = features.mean(dim=2)

        # RNN
        features = features.permute(0, 2, 1)

        rnn_out, _ = self.rnn(features)
        h, attn_weights = self.attention(rnn_out)
        z = self.projection_head(h)

        return h, z
