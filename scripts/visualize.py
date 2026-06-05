"""Project a library's embeddings to 3D (t-SNE) and save an interactive map.

Run after embed_library.py. Writes:
    <out>/projection.csv   track_id, x, y, z + metadata (reused by the app)
    <out>/tsne_3d.html     interactive plotly scatter, colored by genre or BPM

Usage:
    python scripts/visualize.py --embeddings embeddings_output
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from groove.viz.projection import project


def main():
    parser = argparse.ArgumentParser(description="3D t-SNE map of a library.")
    parser.add_argument("--embeddings", default="embeddings_output",
                        help="Dir with embeddings.npy + tracks.csv.")
    parser.add_argument("--dims", type=int, default=3, choices=[2, 3])
    args = parser.parse_args()

    root = Path(args.embeddings)
    emb = np.load(root / "embeddings.npy")
    tracks = pd.read_csv(root / "tracks.csv", dtype={"track_id": str})
    print(f"Projecting {len(emb)} tracks -> {args.dims}D t-SNE...")

    coords = project(emb, n_components=args.dims)
    axes = ["x", "y", "z"][:args.dims]
    df = tracks.copy()
    for i, ax in enumerate(axes):
        df[ax] = coords[:, i]
    df.to_csv(root / "projection.csv", index=False)

    color = "genre" if "genre" in df.columns and df["genre"].notna().any() else "bpm"
    hover = "display_name" if "display_name" in df.columns else "track_id"
    scatter = px.scatter_3d if args.dims == 3 else px.scatter
    fig = scatter(df, *axes, color=color, hover_name=hover,
                  title="Groove map", opacity=0.7)
    fig.update_traces(marker=dict(size=4))
    fig.write_html(root / "tsne_3d.html")
    print(f"Saved projection.csv + tsne_3d.html -> {root}")


if __name__ == "__main__":
    main()
