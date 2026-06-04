"""Brute-force cosine nearest-neighbour index over track embeddings.

The vectors are L2-normalized, so cosine similarity is just a dot product. For
libraries up to ~100k tracks a single matrix multiply is instant, so we don't
need an approximate index (faiss & co.).
"""
import numpy as np
import pandas as pd


class EmbeddingIndex:
    def __init__(self, embeddings, track_ids):
        self.emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.track_ids = list(track_ids)
        self._row = {t: i for i, t in enumerate(self.track_ids)}

    def __len__(self):
        return len(self.track_ids)

    def search(self, queries, k, exclude_rows=None):
        """Top-k index rows for each query vector.

        ``queries`` is [M, D] (L2-normalized). ``exclude_rows[i]`` optionally
        masks one index row per query (to drop the query track itself when
        searching within its own library). Returns (rows [M, k], sims [M, k]),
        each row sorted by decreasing similarity.
        """
        sims = np.asarray(queries, dtype=np.float32) @ self.emb.T
        if exclude_rows is not None:
            sims[np.arange(sims.shape[0]), exclude_rows] = -np.inf
        k = min(k, sims.shape[1])
        top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
        rows = np.take_along_axis(top, order, axis=1)
        return rows, np.take_along_axis(sims, rows, axis=1)

    def neighbors(self, track_id, k=10):
        """(neighbor_track_ids, sims) for one track, excluding itself."""
        r = self._row[track_id]
        rows, sims = self.search(self.emb[r:r + 1], k,
                                 exclude_rows=np.array([r]))
        return [self.track_ids[i] for i in rows[0]], sims[0]

    @classmethod
    def load(cls, emb_path, meta_path):
        """Build an index from embeddings.npy + tracks.csv (returns index, meta)."""
        emb = np.load(emb_path)
        meta = pd.read_csv(meta_path, dtype={"track_id": str})
        return cls(emb, meta["track_id"].tolist()), meta
