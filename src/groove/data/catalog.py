"""Build the track catalog: one row per track with (track_id, source_path, split).

The catalog decouples *where the audio comes from* from the preprocessing step.
Two builders are provided:

- ``build_catalog_from_folder``: the general path. Point ``audio_dir`` at any
  folder of audio files (nested ok); ids are derived from the filename, the
  split is random. This is what lets anyone train on their own library.
- ``build_catalog_from_fma``: FMA-specific, using its metadata to filter by
  duration and genre.
"""
import ast
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")


def _make_track_id(path, root):
    """Readable, collision-safe id: filename slug + short hash of the rel path."""
    rel = path.relative_to(root)
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:40]
    digest = hashlib.sha1(str(rel).encode()).hexdigest()[:8]
    return f"{slug}_{digest}" if slug else digest


def _assign_split(df, val_ratio, test_ratio, seed):
    """Add a 'split' column with random train/val/test labels.

    Uses a shuffled permutation sliced by integer counts, which stays well
    defined for any catalog size (down to a handful of tracks).
    """
    n = len(df)
    perm = np.random.default_rng(seed).permutation(n)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    split = np.array(["train"] * n, dtype=object)
    split[perm[:n_test]] = "test"
    split[perm[n_test:n_test + n_val]] = "val"
    df = df.copy()
    df["split"] = split
    return df


def _save(df, cfg):
    path = Path(cfg.catalog_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Catalog: {len(df)} tracks {df['split'].value_counts().to_dict()} -> {path}")
    return df


def build_catalog_from_folder(cfg, seed=314):
    """Catalog every audio file under ``audio_dir`` with a random split."""
    root = Path(cfg.audio_dir)
    files = [p for p in root.glob("**/*")
             if p.suffix.lower() in AUDIO_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"No audio files found under {root}.")

    df = pd.DataFrame([{
        "track_id": _make_track_id(p, root),
        "source_path": str(p),
        "display_name": p.stem,
    } for p in files]).drop_duplicates("track_id").reset_index(drop=True)

    df = _assign_split(df, cfg.val_ratio, cfg.test_ratio, seed)
    return _save(df, cfg)


# --- FMA-specific mode ------------------------------------------------------
MIN_DURATION = 30
MAX_DURATION = 600
BANNED_GENRES = ["Spoken", "Field Recordings"]


def _banned_genre_ids(genres_csv):
    genres = pd.read_csv(genres_csv, index_col=0)
    ids = []
    for name in BANNED_GENRES:
        match = genres[genres["title"].str.contains(name, case=False, na=False)]
        if len(match):
            ids.append(int(match.index[0]))
    return ids


def _is_banned(genre_str, banned_ids):
    try:
        return not set(ast.literal_eval(genre_str)).isdisjoint(banned_ids)
    except (ValueError, SyntaxError):
        return False


def build_catalog_from_fma(cfg, seed=314):
    """Catalog FMA tracks present on disk, filtered by duration and genre."""
    meta = Path(cfg.fma_metadata_dir)
    tracks = pd.read_csv(meta / "tracks.csv", index_col=0, header=[0, 1])
    df = tracks["track"][["duration", "genres_all", "genre_top"]].copy()

    df = df[(df["duration"] >= MIN_DURATION) & (df["duration"] <= MAX_DURATION)]
    banned = _banned_genre_ids(meta / "genres.csv")
    df = df[~df["genres_all"].apply(lambda g: _is_banned(g, banned))]

    # Keep only metadata entries whose audio file is actually on disk.
    root = Path(cfg.audio_dir)
    id_to_path = {}
    for p in root.glob("**/*"):
        if p.suffix.lower() in AUDIO_EXTENSIONS:
            try:
                id_to_path[int(p.stem)] = p
            except ValueError:
                continue

    rows = []
    for track_id, row in df.iterrows():
        path = id_to_path.get(int(track_id))
        if path is not None:
            rows.append({
                "track_id": f"{int(track_id):06d}",
                "source_path": str(path),
                "display_name": f"{int(track_id):06d}",
                "genre": row["genre_top"],
            })

    cat = pd.DataFrame(rows).reset_index(drop=True)
    cat = _assign_split(cat, cfg.val_ratio, cfg.test_ratio, seed)
    return _save(cat, cfg)
