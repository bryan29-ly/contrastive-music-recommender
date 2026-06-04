import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross-entropy loss (SimCLR).

    A low temperature (~0.1-0.2) sharpens the similarity distribution, which
    suits fine-grained retrieval better than the looser 0.5 default.
    """

    def __init__(self, temperature=0.15):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)
        sim = (z @ z.T) / self.temperature

        # Exclude self-similarity from the denominator.
        diag = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(diag, float("-inf"))

        # The positive of i is its counterpart: i <-> i + batch_size.
        targets = torch.cat([
            torch.arange(batch_size) + batch_size,
            torch.arange(batch_size),
        ]).to(z.device)

        return F.cross_entropy(sim, targets)
