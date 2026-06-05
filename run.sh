#!/usr/bin/env bash
#
# One command to get recommendations on your own library.
#
#   1. Put your audio files (mp3, flac, wav) in data/raw/
#   2. Run:  ./run.sh
#
# First run: builds the catalog, extracts segments, embeds the library with the
# pretrained model, and builds the 3D map, then opens the app. Later runs detect
# the existing embeddings and open the app directly.
#
set -euo pipefail
cd "$(dirname "$0")"

EMB_DIR="embeddings_output"

if [ ! -f "$EMB_DIR/embeddings.npy" ]; then
  echo "No library embeddings yet — setting up (this happens only once)."

  if [ -z "$(find data/raw -type f \( -iname '*.mp3' -o -iname '*.flac' -o -iname '*.wav' \) 2>/dev/null | head -n 1)" ]; then
    echo
    echo "No audio found in data/raw/."
    echo "Add your tracks (mp3, flac, wav) there, then run ./run.sh again."
    exit 1
  fi

  echo "[1/4] Building catalog..."
  uv run scripts/build_catalog.py
  echo "[2/4] Extracting segments..."
  uv run scripts/preprocess.py --resume
  echo "[3/4] Embedding library (downloads the model on first run)..."
  uv run scripts/embed_library.py
  echo "[4/4] Building 3D map..."
  uv run scripts/visualize.py --embeddings "$EMB_DIR"
else
  echo "Found existing embeddings — opening the app."
fi

echo "Launching the recommender..."
uv run streamlit run apps/streamlit_app.py -- --embeddings "$EMB_DIR"
