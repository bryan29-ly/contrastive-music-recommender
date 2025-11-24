import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """Compute the contrastive loss between two views of the same batch

        Args:
            z_i, z_j : tensors [Batch_size, Dim]
        """
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)

        # Similarity matrix
        sim_matrix = torch.mm(z, z.T) / self.temperature

        # Hide the diagonal (don't compare an image with itself)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix.masked_fill_(mask, -9e15)  # Softmax give 0

        # Create labels (the partner of i is i + batch_size)
        labels = torch.cat([
            torch.arange(batch_size) + batch_size,
            torch.arange(batch_size)
        ], dim=0).to(z.device)

        # Compute the loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
