"""Interactive groove-recommendation demo.

Pick a track, get its most rhythmically-similar tracks, and see where they sit on
the 3D groove map. Expects the artifacts produced by embed_library.py and
visualize.py in the same directory.

Usage:
    streamlit run apps/streamlit_app.py -- --embeddings embeddings_output
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from groove.retrieval.index import EmbeddingIndex
from groove.retrieval.matcher import recommend


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="embeddings_output")
    # streamlit passes its own argv; ignore anything it doesn't know.
    return parser.parse_known_args(sys.argv[1:])[0]


@st.cache_resource
def load(root):
    root = Path(root)
    index, meta = EmbeddingIndex.load(root / "embeddings.npy", root / "tracks.csv")
    proj_path = root / "projection.csv"
    proj = pd.read_csv(proj_path, dtype={"track_id": str}) if proj_path.exists() else None
    return index, meta, proj


def main():
    args = _parse_args()
    st.set_page_config(page_title="Groove recommender", layout="wide")
    st.title("🎚️ Groove-based track recommender")

    index, meta, proj = load(args.embeddings)
    label_col = "display_name" if "display_name" in meta.columns else "track_id"
    labels = dict(zip(meta["track_id"], meta[label_col]))

    with st.sidebar:
        st.header("Query")
        track_id = st.selectbox("Track", meta["track_id"],
                                format_func=lambda t: labels.get(t, t))
        k = st.slider("Recommendations", 1, 20, 8)
        use_bpm = st.checkbox("Limit to similar BPM (DJ mode)")
        bpm_tol = st.slider("± BPM", 1, 30, 6) if use_bpm else None
        if "source_path" in meta.columns:
            src = meta.set_index("track_id").loc[track_id, "source_path"]
            if isinstance(src, str) and Path(src).exists():
                st.audio(src)

    recs = recommend(index, meta, track_id, k=k, bpm_tolerance=bpm_tol)
    col_recs, col_map = st.columns([1, 1.4])

    with col_recs:
        st.subheader("Most similar")
        st.dataframe(recs, use_container_width=True, hide_index=True)

    with col_map:
        if proj is None:
            st.info("Run scripts/visualize.py to enable the 3D map.")
        else:
            df = proj.copy()
            highlight = np.where(df["track_id"] == track_id, "query",
                                 np.where(df["track_id"].isin(recs["track_id"]),
                                          "recommended", "library"))
            df["role"] = highlight
            color = "role"
            fig = px.scatter_3d(
                df, x="x", y="y", z="z", color=color,
                hover_name=label_col if label_col in df.columns else "track_id",
                color_discrete_map={"library": "#cccccc", "recommended": "#ff7f0e",
                                    "query": "#d62728"}, opacity=0.8)
            fig.update_traces(marker=dict(size=4))
            fig.update_layout(height=650, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
