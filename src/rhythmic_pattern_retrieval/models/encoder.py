import torch.nn as nn
from torchvision.models import resnet18


class RhythmicEncoder(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()

        # Load Resnet18 with no weights
        self.backbone = resnet18(weights=None)

        # Modify the first layer to accept 1 channel (Spectrogram)
        self.backbone.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Remove the classification layer
        self.backbone.fc = nn.Identity()

        # Projection for Contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return h, z
