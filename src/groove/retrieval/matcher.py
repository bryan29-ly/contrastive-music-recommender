"""Readable top-k recommendations on top of an EmbeddingIndex."""
import pandas as pd

_META_COLS = ("display_name", "bpm", "genre")


def recommend(index, meta, track_id, k=10, bpm_tolerance=None):
    """Return the k most similar tracks to ``track_id`` as a DataFrame.

    ``meta`` is the tracks table (track_id + optional display_name/bpm/genre).
    ``bpm_tolerance`` (optional, in BPM) keeps only neighbours within +/- that of
    the query, for beatmatched DJ transitions; since it filters *after*
    retrieval, we over-fetch to still return up to k results.
    """
    info = meta.set_index("track_id")
    cols = [c for c in _META_COLS if c in info.columns]
    fetch = k * 5 if bpm_tolerance else k
    ids, sims = index.neighbors(track_id, k=fetch)

    q_bpm = info.loc[track_id, "bpm"] if bpm_tolerance and "bpm" in cols else None
    rows = []
    for tid, sim in zip(ids, sims):
        row = info.loc[tid]
        if bpm_tolerance is not None and abs(row["bpm"] - q_bpm) > bpm_tolerance:
            continue
        rows.append({"track_id": tid, "similarity": round(float(sim), 4),
                     **{c: row[c] for c in cols}})
        if len(rows) >= k:
            break
    return pd.DataFrame(rows)
