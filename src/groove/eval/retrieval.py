"""Offline retrieval-quality metrics on a held-out split.

These are the heavier, global-nearest-neighbour metrics we keep out of the
training loop. Embeddings are L2-normalized, so cosine = dot product.
"""
import numpy as np

from groove.retrieval.index import EmbeddingIndex


def recall_same_track(seg_df, seg_emb, ks=(1, 5, 10)):
    """Segment-level: does a segment retrieve another segment of the same track?

    Restricted to segments whose track has at least one sibling. Returns a dict
    {k: recall@k}. A coherence sanity check, not the end goal.
    """
    tracks = seg_df["track_id"].to_numpy()
    counts = seg_df["track_id"].value_counts()
    has_sibling = np.isin(tracks, counts[counts > 1].index)
    if not has_sibling.any():
        return {k: float("nan") for k in ks}

    idx = EmbeddingIndex(seg_emb, list(range(len(seg_emb))))
    q_rows = np.where(has_sibling)[0]
    rows, _ = idx.search(seg_emb[q_rows], max(ks), exclude_rows=q_rows)
    hit = tracks[rows] == tracks[q_rows][:, None]
    return {k: float(hit[:, :k].any(axis=1).mean()) for k in ks}


def genre_purity(emb, genres, k=10):
    """Track-level: fraction of a track's k nearest tracks sharing its genre.

    Returns (purity@k, random_baseline). Baseline = chance two random tracks
    share a genre (sum of squared genre frequencies); purity well above it means
    the space groups musically. Only tracks with a known genre are scored.
    """
    g = np.asarray(genres, dtype=object)
    keep = np.array([x is not None and x == x for x in g])  # drop None/NaN
    if keep.sum() < 2:
        return float("nan"), float("nan")
    emb, g = emb[keep], g[keep]

    idx = EmbeddingIndex(emb, list(range(len(emb))))
    rows, _ = idx.search(emb, k, exclude_rows=np.arange(len(emb)))
    purity = float((g[rows] == g[:, None]).mean())

    _, cnt = np.unique(g, return_counts=True)
    baseline = float(((cnt / cnt.sum()) ** 2).sum())
    return purity, baseline


def bpm_coherence(emb, bpms, seed=0):
    """Track-level: mean |BPM(query) - BPM(nearest)| vs a random-pairing baseline.

    Returns (nn_diff, random_diff, ratio). ratio < 1 means nearest neighbours
    are closer in tempo than chance -> the space is partly tempo-organized.
    """
    bpms = np.asarray(bpms, dtype="float32")
    idx = EmbeddingIndex(emb, list(range(len(emb))))
    rows, _ = idx.search(emb, 1, exclude_rows=np.arange(len(emb)))
    nn_diff = float(np.abs(bpms - bpms[rows[:, 0]]).mean())
    perm = np.random.default_rng(seed).permutation(len(bpms))
    rand_diff = float(np.abs(bpms - bpms[perm]).mean())
    ratio = nn_diff / rand_diff if rand_diff else float("nan")
    return nn_diff, rand_diff, ratio


def alignment_uniformity(seg_df, seg_emb, max_uniform=4000, seed=0):
    """Wang & Isola (2020) proxies on the split (segment level).

    alignment : mean squared distance between two random segments of the same
                (multi-segment) track. uniformity: log-mean Gaussian potential
                over segment embeddings (subsampled for cost). Both: lower is
                better.
    """
    rng = np.random.default_rng(seed)
    groups = seg_df.groupby("track_id", sort=False).indices
    aligns = []
    for pos in groups.values():
        if len(pos) > 1:
            a, b = rng.choice(pos, size=2, replace=False)
            aligns.append(float(((seg_emb[a] - seg_emb[b]) ** 2).sum()))
    align = float(np.mean(aligns)) if aligns else float("nan")

    e = seg_emb
    if len(e) > max_uniform:
        e = e[rng.choice(len(e), max_uniform, replace=False)]
    sq = np.maximum(2.0 - 2.0 * (e @ e.T), 0.0)
    off = ~np.eye(len(e), dtype=bool)
    uniform = float(np.log(np.exp(-2.0 * sq[off]).mean()))
    return align, uniform
