"""Embed a preprocessed library into one vector per track.

Run after build_catalog.py + preprocess.py. Loads a trained checkpoint, encodes
every segment in the manifest, and writes (row-aligned):
    <out>/embeddings.npy   float32 [N, D], L2-normalized
    <out>/tracks.csv       track_id, bpm[, display_name, genre]

Usage:
    python scripts/embed_library.py --checkpoint models_output/<run>/best.pt
    python scripts/embed_library.py --checkpoint ... --split test   # eval subset
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from groove.config import load_config
from groove.retrieval.embed import (DEFAULT_CHECKPOINT, embed_library,
                                     load_encoder)
from groove.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(
        description="Embed a library into vectors.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Local .pt path or a Hugging Face repo id (auto-downloaded).")
    parser.add_argument("--split", default=None,
                        help="Restrict to one split (e.g. test).")
    parser.add_argument("--out", default="embeddings_output",
                        help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    # local data paths (the library being embedded)
    paths = load_config("data")
    device = get_device()
    # audio params from training
    model, mel, cfg = load_encoder(args.checkpoint, device)

    manifest = pd.read_csv(Path(paths.segments_dir) /
                           "manifest.csv", dtype={"track_id": str})
    if args.split:
        manifest = manifest[manifest["split"] ==
                            args.split].reset_index(drop=True)
    print(f"device={device.type} | embedding {manifest['track_id'].nunique()} tracks "
          f"({len(manifest)} segments)" + (f" [split={args.split}]" if args.split else ""))

    ids, emb, bpm = embed_library(
        manifest, paths.segments_dir, model, mel, cfg, device, args.batch_size)

    tracks = pd.DataFrame({"track_id": ids, "bpm": bpm})
    if Path(paths.catalog_path).exists():  # enrich with display_name / genre
        cat = pd.read_csv(paths.catalog_path, dtype={"track_id": str})
        keep = ["track_id"] + [c for c in ("display_name", "genre", "source_path")
                               if c in cat.columns]
        tracks = tracks.merge(cat[keep], on="track_id", how="left")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb)
    tracks.to_csv(out / "tracks.csv", index=False)
    print(f"Saved embeddings {emb.shape} -> {out}/embeddings.npy + tracks.csv")


if __name__ == "__main__":
    main()
