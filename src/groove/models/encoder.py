import torch.nn as nn

from groove.models.blocks import POOLINGS, ResBlock


class RhythmicEncoder(nn.Module):
    """ResNet encoder tuned for rhythm: asymmetric strides shrink the frequency
    axis while preserving temporal resolution up to the pooling head.

    Returns (h, z): ``h`` is the representation used for retrieval, ``z`` the
    projection used only by the contrastive loss.
    """

    def __init__(self, cfg):
        super().__init__()
        c = cfg.encoder.base_channels

        # Stem: shrink frequency (stride 2), keep time (stride 1).
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1),
        )
        # Backbone: layers 3-4 use stride (2, 1) -> reduce frequency only, so the
        # rhythmic (time) resolution survives to the top of the network.
        self.layer1 = self._make_layer(c, c, (1, 1))
        self.layer2 = self._make_layer(c, c * 2, (2, 2))
        self.layer3 = self._make_layer(c * 2, c * 4, (2, 1))
        self.layer4 = self._make_layer(c * 4, c * 8, (2, 1))

        self.pool = POOLINGS[cfg.encoder.pooling](c * 8)
        self.embedding_dim = self.pool.output_dim

        # Projection head (SimCLR): used only for the contrastive loss.
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, cfg.projection.hidden_dim, bias=False),
            nn.BatchNorm1d(cfg.projection.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.projection.hidden_dim, cfg.projection.output_dim, bias=False),
            nn.BatchNorm1d(cfg.projection.output_dim),
        )

    def _make_layer(self, in_ch, out_ch, stride):
        return nn.Sequential(
            ResBlock(in_ch, out_ch, stride),
            ResBlock(out_ch, out_ch, (1, 1)),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.mean(dim=2)        # average over remaining frequency -> [B, C, T]
        h = self.pool(x)         # [B, embedding_dim]
        z = self.projection(h)   # [B, projection_dim]
        return h, z
