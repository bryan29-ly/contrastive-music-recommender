"""Build the track catalog (track_id, source_path, split).

Default mode scans a folder of audio files (your own library). FMA mode uses
the FMA metadata for duration/genre filtering.

Usage:
    python scripts/build_catalog.py               # folder mode (data.audio_dir)
    python scripts/build_catalog.py --source fma  # FMA metadata mode
"""
import argparse

from groove.config import load_config
from groove.data.catalog import build_catalog_from_folder, build_catalog_from_fma

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the track catalog.")
    parser.add_argument("--source", choices=["folder", "fma"], default="folder")
    args = parser.parse_args()

    cfg = load_config("data")
    if args.source == "fma":
        build_catalog_from_fma(cfg)
    else:
        build_catalog_from_folder(cfg)
