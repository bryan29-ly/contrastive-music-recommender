"""Evaluate retrieval quality on a held-out split (default: test).

Embeds the split and reports:
    recall@k (same-track)  segment-level coherence sanity check
    genre purity@k         do a track's nearest tracks share its genre? (FMA)
    bpm coherence          are neighbours closer in tempo than chance?
    alignment / uniformity Wang & Isola proxies on the split

Usage:
    python scripts/evaluate.py --checkpoint models_output/<run>/best.pt
"""
import argparse
from pathlib import Path

import pandas as pd

from groove.config import load_config
from groove.eval.retrieval import (alignment_uniformity, bpm_coherence,
                                   genre_purity, recall_same_track)
from groove.retrieval.embed import (DEFAULT_CHECKPOINT, embed_segments,
                                     load_encoder, pool_to_tracks)
from groove.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Local .pt path or a Hugging Face repo id (auto-downloaded).")
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=10,
                        help="k for recall/purity.")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    paths = load_config("data")
    device = get_device()
    model, mel, cfg = load_encoder(args.checkpoint, device)

    manifest = pd.read_csv(Path(paths.segments_dir) /
                           "manifest.csv", dtype={"track_id": str})
    manifest = manifest[manifest["split"] == args.split].reset_index(drop=True)
    print(f"device={device.type} | split={args.split} | "
          f"{manifest['track_id'].nunique()} tracks ({len(manifest)} segments)\n")

    seg_df, seg_emb = embed_segments(
        manifest, paths.segments_dir, model, mel, cfg, device, args.batch_size)
    track_ids, track_emb, bpms = pool_to_tracks(seg_df, seg_emb)

    # genre comes from the catalog (FMA); absent for generic folders.
    genres = None
    if Path(paths.catalog_path).exists():
        cat = pd.read_csv(paths.catalog_path, dtype={"track_id": str})
        if "genre" in cat.columns:
            gmap = dict(zip(cat["track_id"], cat["genre"]))
            genres = [gmap.get(t) for t in track_ids]

    recall = recall_same_track(seg_df, seg_emb, ks=(1, 5, args.k))
    align, uniform = alignment_uniformity(seg_df, seg_emb)
    nn_bpm, rand_bpm, bpm_ratio = bpm_coherence(track_emb, bpms)

    print("== retrieval quality ==")
    print("  recall@k (same-track): " +
          "  ".join(f"@{k}={v:.3f}" for k, v in recall.items()))
    if genres is not None:
        purity, base = genre_purity(track_emb, genres, k=args.k)
        print(
            f"  genre purity@{args.k}:    {purity:.3f}  (random baseline {base:.3f})")
    else:
        print("  genre purity:          skipped (no genre in catalog)")
    print(
        f"  bpm coherence:         nn={nn_bpm:.1f}  random={rand_bpm:.1f}  ratio={bpm_ratio:.3f}")
    print(f"  alignment / uniformity:{align:.4f} / {uniform:.4f}")


if __name__ == "__main__":
    main()
