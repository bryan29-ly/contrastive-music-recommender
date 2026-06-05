"""Interactive groove-recommendation demo.

Pick a track from an embedded library, get its most rhythmically-similar tracks,
listen to the query *and* every recommendation, and see them on a 3D groove map.

One app, two audiences:
  - DJs browse a collection embedded with embed_library.py.
  - Whoever trained the model gets a "Test set only" switch -- shown only when the
    training manifest is found locally -- to demo on held-out, never-seen tracks.

Expects embeddings.npy + tracks.csv (+ optional projection.csv) produced by
embed_library.py / visualize.py.

Usage:
    streamlit run apps/streamlit_app.py -- --embeddings embeddings_output
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from groove.retrieval.index import EmbeddingIndex
from groove.retrieval.matcher import recommend

QUERY_COLOR = "#ef476f"
REC_COLOR = "#ff8c42"
LIB_COLOR = "#c9c9d4"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="embeddings_output")
    parser.add_argument("--manifest", default=None,
                        help="Training manifest; enables the 'test set only' filter.")
    return parser.parse_known_args(sys.argv[1:])[0]


def _default_manifest():
    """Locate the training manifest so the test-split filter self-enables.

    Split labels live in the manifest, not the embeddings -- so only someone with
    the training data (i.e. who trained the model) ever sees the filter. DJs who
    just embed their own collection have no manifest and never see it.
    """
    try:
        from groove.config import load_config
        path = Path(load_config("data").segments_dir) / "manifest.csv"
        return str(path) if path.exists() else None
    except Exception:
        return None


@st.cache_resource
def load(root, manifest_path):
    root = Path(root)
    emb = np.load(root / "embeddings.npy")
    meta = pd.read_csv(root / "tracks.csv", dtype={"track_id": str})
    proj_path = root / "projection.csv"
    proj = pd.read_csv(proj_path, dtype={
                       "track_id": str}) if proj_path.exists() else None

    if manifest_path and Path(manifest_path).exists():
        man = pd.read_csv(manifest_path, dtype={
                          "track_id": str}, usecols=["track_id", "split"])
        splits = man.drop_duplicates("track_id").set_index("track_id")["split"]
        meta = meta.merge(splits.rename("split"),
                          left_on="track_id", right_index=True, how="left")

    full_index = EmbeddingIndex(emb, meta["track_id"].tolist())
    return emb, meta, proj, full_index


def _subset(emb, meta, proj, mask):
    """Restrict index / metadata / projection to a boolean track mask."""
    view = meta[mask].reset_index(drop=True)
    index = EmbeddingIndex(emb[mask.to_numpy()], view["track_id"].tolist())
    proj_view = proj[proj["track_id"].isin(
        view["track_id"])] if proj is not None else None
    return index, view, proj_view


def _audio(meta_idx, track_id):
    src = meta_idx.loc[track_id].get("source_path")
    if isinstance(src, str) and Path(src).exists():
        st.audio(src)
    else:
        st.caption("_audio unavailable_")


def _chips(row, cols):
    bits = []
    if "bpm" in cols and pd.notna(row.get("bpm")):
        bits.append(f"{float(row['bpm']):.0f} BPM")
    if "genre" in cols and isinstance(row.get("genre"), str):
        bits.append(row["genre"])
    return " · ".join(bits)


def groove_map(proj_view, labels, query_id, rec_ids, hover_col):
    """3D scatter of the library with the query and its recommendations on top."""
    role = np.where(proj_view["track_id"] == query_id, "query",
                    np.where(proj_view["track_id"].isin(rec_ids), "recommended", "library"))
    proj_view = proj_view.assign(role=role)

    fig = go.Figure()
    for name, color, size in [("library", LIB_COLOR, 2.5),
                              ("recommended", REC_COLOR, 6.0),
                              ("query", QUERY_COLOR, 9.0)]:
        d = proj_view[proj_view["role"] == name]
        if d.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=d["x"], y=d["y"], z=d["z"], mode="markers", name=name,
            marker=dict(size=size, color=color,
                        opacity=0.45 if name == "library" else 0.95),
            hovertext=d["track_id"].map(
                labels) if hover_col else d["track_id"],
            hoverinfo="text"))
    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=0.02, x=0.5,
                    xanchor="center", title_text=""),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False)))
    return fig


def main():
    args = _parse_args()
    manifest_path = args.manifest or _default_manifest()
    st.set_page_config(page_title="Groove Recommender",
                       page_icon="🎚️", layout="wide")

    emb, meta, proj, full_index = load(args.embeddings, manifest_path)
    has_splits = "split" in meta.columns and (meta["split"] == "test").any()

    st.title("🎚️ Groove-based Track Recommender")
    st.caption("Self-supervised rhythm embeddings (SimCLR, trained from scratch) — "
               "finds tracks that **groove alike**.")

    # Filter first, because it decides which library the rest of the UI runs on.
    with st.sidebar:
        st.header("Query")
        test_only = (has_splits and
                     st.toggle("Test set only", value=False,
                               help="Keep tracks the model never saw during "
                                    "training."))

    if test_only:
        index, view, proj_view = _subset(
            emb, meta, proj, meta["split"] == "test")
    else:
        index, view, proj_view = full_index, meta, proj

    label_col = "display_name" if "display_name" in view.columns else "track_id"
    labels = {t: str(n) for t, n in zip(view["track_id"], view[label_col])}
    view_idx = view.set_index("track_id")

    with st.sidebar:
        track_id = st.selectbox("Track", list(view["track_id"]),
                                format_func=lambda t: labels.get(t, t))
        k = st.slider("Recommendations", 1, 20, 8)
        dj_mode = st.toggle("DJ mode — similar BPM only")
        bpm_tol = st.slider("± BPM", 1, 30, 6) if dj_mode else None

        st.divider()
        st.markdown("**Now playing**")
        st.write(labels[track_id])
        chips = _chips(view_idx.loc[track_id], view.columns)
        if chips:
            st.caption(chips)
        _audio(view_idx, track_id)
        st.caption(f"{len(view):,} tracks in library" +
                   (" · test split" if test_only else ""))

    recs = recommend(index, view, track_id, k=k, bpm_tolerance=bpm_tol)
    col_recs, col_map = st.columns([1, 1.3], gap="large")

    with col_recs:
        st.subheader("Most similar")
        if recs.empty:
            st.info(
                "No match within this BPM tolerance — widen it or turn off DJ mode.")
        for rank, row in enumerate(recs.itertuples(index=False), 1):
            rid = row.track_id
            with st.container(border=True):
                head, score = st.columns([0.65, 0.35])
                head.markdown(f"**{rank}. {labels.get(rid, rid)}**")
                chips = _chips(view_idx.loc[rid], view.columns)
                if chips:
                    head.caption(chips)
                score.metric("similarity", f"{row.similarity:.3f}")
                _audio(view_idx, rid)

    with col_map:
        st.subheader("Groove map")
        if proj_view is None:
            st.info("Run `scripts/visualize.py` to generate the 3D map.")
        else:
            fig = groove_map(proj_view, labels, track_id, set(recs["track_id"]),
                             hover_col=label_col)
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("About this demo"):
        st.markdown(
            "Each track is embedded by a convolutional encoder trained with "
            "**SimCLR** contrastive learning on short audio segments — "
            "no genre or BPM labels are used at training time. Tracks that share a "
            "**rhythmic feel** end up close in the embedding space; recommendations "
            "are exact cosine nearest neighbours. *DJ mode* additionally constrains "
            "neighbours to a tempo window for beatmatched transitions.")


if __name__ == "__main__":
    main()
