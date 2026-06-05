"""Offline preprocessing: extract rhythmically-salient audio segments.

Reads the catalog (see scripts/build_catalog.py), and for every track selects a
few short windows (audio/segmentation.py), estimates its BPM, and stores each
window as a lossless int16 FLAC. Mel spectrograms are computed on the GPU at
train time, so we keep only compact waveform segments on disk.

Usage:
    python scripts/preprocess.py                 # all catalog tracks
    python scripts/preprocess.py --limit 50      # smoke test on 50 tracks
    python scripts/preprocess.py --resume        # skip already-done tracks
"""
import argparse
import multiprocessing
from pathlib import Path

import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm

from groove.config import load_config
from groove.audio.io import load_audio
from groove.audio.segmentation import estimate_bpm, select_segments


def process_track(track_id, split, source_path, cfg, out_dir):
    """Decode one track, select segments, write FLAC files, return manifest rows."""
    sr = cfg.sample_rate
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    try:
        wav_np = load_audio(source_path, target_sr=sr, mono=True).numpy()
        if len(wav_np) < cfg.min_track_seconds * sr:
            return []  # too short to be useful

        bpm = estimate_bpm(wav_np, sr)
        segments = select_segments(
            wav_np, sr, cfg.segment_seconds, cfg.n_segments_per_track)

        rows = []
        for idx, (start, end) in enumerate(segments):
            seg_path = split_dir / f"{track_id}_{idx}.flac"
            sf.write(seg_path, wav_np[start:end], sr,
                     subtype="PCM_16", format="FLAC")
            rows.append({
                "track_id": track_id,
                "split": split,
                "segment_idx": idx,
                "segment_path": str(seg_path.relative_to(out_dir)),
                "bpm": round(bpm, 2),
                "sample_rate": sr,
                "source_path": source_path,
            })
        return rows
    except Exception as e:  # one bad file shouldn't kill the whole job
        print(f"Error on {Path(source_path).name}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Extract audio segments.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N tracks (smoke test).")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Parallel CPU jobs (default: cpu_count - 1).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tracks already present in manifest.csv.")
    args = parser.parse_args()

    cfg = load_config("audio", "data")
    out_dir = Path(cfg.segments_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    n_jobs = args.jobs or max(1, multiprocessing.cpu_count() - 1)

    catalog = pd.read_csv(cfg.catalog_path, dtype={"track_id": str})
    print(f"Loaded catalog: {len(catalog)} tracks.")

    existing = None
    if args.resume and manifest_path.exists():
        existing = pd.read_csv(manifest_path, dtype={"track_id": str})
        done = set(existing["track_id"])
        catalog = catalog[~catalog["track_id"].isin(done)]
        print(f"Resume: {len(done)} tracks already done, {len(catalog)} remaining.")

    if args.limit:
        catalog = catalog.iloc[:args.limit]

    print(f"Processing {len(catalog)} tracks with {n_jobs} jobs...")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_track)(r.track_id, r.split, r.source_path, cfg, out_dir)
        for r in tqdm(catalog.itertuples(index=False), total=len(catalog),
                      desc="Preprocessing", unit="track")
    )

    rows = [r for track_rows in results for r in track_rows]
    df = pd.DataFrame(rows)
    if existing is not None:
        df = pd.concat([existing, df], ignore_index=True) if len(df) else existing
    df.to_csv(manifest_path, index=False)

    n_tracks = df["track_id"].nunique() if len(df) else 0
    print(f"Done. {len(df)} segments from {n_tracks} tracks -> {manifest_path}")


if __name__ == "__main__":
    main()
