"""Low-dimensional projection of track embeddings for visualization."""
import numpy as np
from openTSNE import TSNE
from sklearn.decomposition import PCA


def project(emb, n_components=3, pca_dim=50, perplexity=30, seed=314, n_jobs=-1):
    """PCA-denoise then t-SNE (openTSNE, multicore) to ``n_components`` dims.

    PCA first speeds up t-SNE and removes noise directions; perplexity is capped
    so it stays valid on small libraries. openTSNE parallelizes the Barnes-Hut
    optimization across ``n_jobs`` cores (``-1`` = all). Reproducible for a fixed
    ``(seed, n_jobs)``. Returns coords [N, n_components].
    """
    x = np.asarray(emb, dtype="float32")
    if pca_dim and min(x.shape) > pca_dim:
        x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    perplexity = min(perplexity, max(5, (len(x) - 1) // 3))
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        initialization="pca",
        negative_gradient_method="bh",  # bh supports 3D; fft is 2D-only
        n_jobs=n_jobs,
        random_state=seed,
        verbose=True,
    )
    return np.asarray(tsne.fit(x))
