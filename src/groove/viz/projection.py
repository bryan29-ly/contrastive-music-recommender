"""Low-dimensional projection of track embeddings for visualization."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def project(emb, n_components=3, pca_dim=50, perplexity=30, seed=314):
    """PCA-denoise then t-SNE to ``n_components`` dims (2 or 3).

    PCA first speeds up t-SNE and removes noise directions. Perplexity is capped
    so it stays valid on small libraries. Returns coords [N, n_components].
    """
    x = np.asarray(emb, dtype="float32")
    if pca_dim and min(x.shape) > pca_dim:
        x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    perplexity = min(perplexity, max(5, (len(x) - 1) // 3))
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                init="pca", random_state=seed)
    return tsne.fit_transform(x)
