import torch.nn as nn

from groove.models.blocks import (AttentivePooling, ResBlock,
                                  SinusoidalPositionalEncoding)

# Total frequency stride of the backbone (stem 2, maxpool 2, layers 2/3/4).
FREQ_STRIDE = 32


class RhythmicEncoder(nn.Module):
    """Log-mel spectrogram -> groove embedding.

    A 2D ResNet strides frequency while leaving time nearly intact, then a
    small transformer gives every remaining frame global context over the crop.
    The convolutions alone reach a temporal receptive field of ~1.3 s, under one
    bar at most tempos, and pooling applied straight after them is
    permutation-invariant; the transformer is what makes a bar-length rhythmic
    pattern representable.

    Returns (h, z): h is the retrieval representation, z the projection used
    only by the contrastive loss.
    """

    def __init__(self, cfg):
        super().__init__()
        c = cfg.encoder.base_channels
        dim = cfg.encoder.embed_dim

        # Stem: shrink frequency (stride 2), keep time (stride 1).
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, 5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=(2, 2), padding=1),
        )
        # Layers 3-4 stride frequency only. Total time stride is 4, leaving
        # ~21.5 frames per second for the transformer to work on.
        self.layer1 = self._make_layer(c, c, (1, 1))
        self.layer2 = self._make_layer(c, c * 2, (2, 2))
        self.layer3 = self._make_layer(c * 2, c * 4, (2, 1))
        self.layer4 = self._make_layer(c * 4, c * 8, (2, 1))

        # Surviving frequency bands are folded into the token rather than
        # averaged: which band carries the energy is what separates a kick from
        # a snare from a hat.
        token_dim = c * 8 * (cfg.n_mels // FREQ_STRIDE)
        self.to_tokens = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, dim),
        )
        self.positions = SinusoidalPositionalEncoding(dim)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                dim, cfg.encoder.n_heads, cfg.encoder.ffn_dim,
                dropout=cfg.encoder.dropout, activation="gelu",
                batch_first=True, norm_first=True,
            ),
            num_layers=cfg.encoder.n_layers,
            norm=nn.LayerNorm(dim),
            enable_nested_tensor=False,
        )
        self.pool = AttentivePooling(dim)
        self.embedding_dim = dim

        # Projection head (SimCLR): used only for the contrastive loss.
        self.projection = nn.Sequential(
            nn.Linear(dim, cfg.projection.hidden_dim, bias=False),
            nn.BatchNorm1d(cfg.projection.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.projection.hidden_dim,
                      cfg.projection.output_dim, bias=False),
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

        # [B, C, F, T] -> [B, T, C*F]: one token per time step.
        x = x.permute(0, 3, 1, 2).flatten(2)
        x = self.positions(self.to_tokens(x))

        h = self.pool(self.temporal(x))  # [B, embed_dim]
        z = self.projection(h)           # [B, projection_dim]
        return h, z
