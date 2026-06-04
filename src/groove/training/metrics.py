import torch
import torch.nn.functional as F


@torch.no_grad()
def representation_metrics(z1, z2):
    """Cheap in-batch diagnostics for a retrieval-oriented contrastive space.

    Everything is computed from the current batch's projection embeddings (no
    global nearest-neighbour search), so it is essentially free to log every
    epoch. ``z1``/``z2`` are the two views of the same segments, shape [B, D].

    Returns a dict:
      align    mean squared distance between positive pairs (Wang & Isola 2020);
               low = positives land close together.
      uniform  log-mean Gaussian potential over the batch (Wang & Isola 2020);
               low/negative = embeddings spread over the hypersphere.
      acc1     in-batch retrieval top-1: how often a view's true partner is its
               nearest neighbour among the batch -> direct retrieval proxy.
      acc5     same, but counting a hit within the 5 nearest neighbours.
      emb_std  mean per-dimension std of the normalized embeddings; collapses
               toward 0 under dimensional collapse (healthy ~ 1/sqrt(D)).
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    n = z1.shape[0]

    align = (z1 - z2).pow(2).sum(dim=1).mean()

    # Uniformity over all distinct pairs. We avoid torch.pdist (unimplemented on
    # MPS) and use squared distances from the Gram matrix: for unit vectors
    # ||a - b||^2 = 2 - 2 a.b. The diagonal (self-pairs) is masked out.
    z = torch.cat([z1, z2], dim=0)
    sq_dist = (2.0 - 2.0 * (z @ z.t())).clamp(min=0.0)
    off_diag = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    uniform = sq_dist[off_diag].mul(-2.0).exp().mean().log()

    # Row i of the view1->view2 similarity matrix should rank partner i first.
    sim = z1 @ z2.t()
    target = torch.arange(n, device=z1.device)
    ranks = sim.argsort(dim=1, descending=True)
    acc1 = (ranks[:, 0] == target).float().mean()
    k = min(5, n)
    acc5 = (ranks[:, :k] == target.unsqueeze(1)).any(dim=1).float().mean()

    emb_std = z.std(dim=0).mean()

    return {"align": align.item(), "uniform": uniform.item(),
            "acc1": acc1.item(), "acc5": acc5.item(), "emb_std": emb_std.item()}
