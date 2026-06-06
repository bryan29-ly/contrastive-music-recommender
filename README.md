# Groove-based Music Recommender

A music recommender that finds tracks with a similar rhythmic feel ("groove"),
using audio embeddings learned by self-supervised contrastive learning (SimCLR).

There are two ways to use this repository:

- **Get recommendations** on your own music library, using a pretrained model.
- **Train your own model** from scratch on your own dataset.

## How it works

The encoder is trained with contrastive learning (SimCLR / NT-Xent) on short
audio segments — no genre or BPM labels are used. An optional BPM filter keeps neighbours within a tempo
window for beatmatched DJ transitions.


## Installation

With [uv](https://github.com/astral-sh/uv) (recommended):

```bash
git clone https://github.com/bryan29-ly/contrastive-music-recommender.git
cd contrastive-music-recommender
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Get recommendations on your own library

One command, using the pretrained model (downloaded automatically on first use).

1. Put your audio files (mp3, flac, wav; nested folders are fine) in `data/raw/`.
2. Run:

```bash
./run.sh
```

The first run builds your catalog, extracts segments, embeds the library, builds
the 3D map, and opens the app. Later runs detect the existing embeddings and open
the app directly.

In the app, pick a track to hear it and its most rhythmically-similar neighbours,
see them on the 3D groove map, and toggle DJ mode to constrain recommendations to
a BPM window.

## Train your own model

**1. Prepare the data.** Put audio in `data/raw/`, then build the catalog and
extract segments:

```bash
uv run scripts/build_catalog.py
uv run scripts/preprocess.py
```

To train on the [FMA](https://github.com/mdeff/fma) dataset instead of a plain
folder, use `build_catalog.py --source fma` (see `configs/data.yaml`).

**2. Compute normalization statistics**

```bash
uv run scripts/compute_norm_stats.py
```

**3. Train.** Settings live in `configs/train.yaml` (batch size, epochs, learning
rate, temperature, etc.):

```bash
uv run scripts/train.py --name my-run
```

Checkpoints are written to `models_output/<timestamp>/` — `best.pt` (best
validation) and `last.pt` (end of the schedule).

**4. Embed and visualize with your checkpoint, then open the app.** Pass your
local checkpoint directly :

```bash
uv run scripts/embed_library.py --checkpoint models_output/<run>/best.pt
uv run scripts/visualize.py --embeddings embeddings_output
uv run streamlit run apps/streamlit_app.py -- --embeddings embeddings_output
```

`--checkpoint` accepts a local path or a Hugging Face repo id interchangeably.
When the training manifest is present, the app also shows a "Test set only"
switch to demo recommendations on tracks the model never saw.

## Project structure

```
configs/        YAML configs (audio, data, model, augmentation, training)
scripts/        CLI entry points (catalog, preprocess, train, embed, visualize)
src/groove/     library code (audio, models, training, retrieval, eval, viz)
apps/           Streamlit recommendation app
run.sh          one-command setup + app for end users
```

## Acknowledgements

Trained and evaluated on the [Free Music Archive (FMA)](https://github.com/mdeff/fma)
dataset.
