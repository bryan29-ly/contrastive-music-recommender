"""Compute the global log-mel mean/std over a sample of training segments.

These standardize the encoder input (stable early training, consistent scale).
Saved to <segments_dir>/norm_stats.json and loaded automatically at train and
inference time.

Usage:
    python scripts/compute_norm_stats.py            # 2000 segments
    python scripts/compute_norm_stats.py --sample 5000
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch

from groove.audio.features import LogMelSpectrogram
from groove.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Compute log-mel norm stats.")
    parser.add_argument("--sample", type=int, default=2000,
                        help="Number of training segments to sample.")
    args = parser.parse_args()

    cfg = load_config("audio", "data")
    seg_dir = Path(cfg.segments_dir)
    df = pd.read_csv(seg_dir / "manifest.csv", dtype={"track_id": str})
    paths = df[df["split"] == "train"]["segment_path"].tolist()
    random.seed(314)
    random.shuffle(paths)
    paths = paths[:args.sample]

    # norm_mean/std are still null here -> raw log-mel, exactly what we measure.
    mel = LogMelSpectrogram(cfg)

    total, s, s2 = 0, 0.0, 0.0
    for rel in paths:
        wav, _ = sf.read(seg_dir / rel, dtype="float32")
        x = mel(torch.from_numpy(wav))
        s += x.sum().item()
        s2 += (x * x).sum().item()
        total += x.numel()

    mean = s / total
    std = max(s2 / total - mean ** 2, 0.0) ** 0.5
    stats = {"mean": mean, "std": std,
             "n_segments": len(paths), "n_values": total}

    out = seg_dir / "norm_stats.json"
    out.write_text(json.dumps(stats, indent=2))
    print(f"log-mel mean={mean:.4f} std={std:.4f} "
          f"(from {len(paths)} segments) -> {out}")


if __name__ == "__main__":
    main()
