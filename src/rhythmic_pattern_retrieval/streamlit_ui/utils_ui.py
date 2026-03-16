import streamlit as st
from pathlib import Path
from rhythmic_pattern_retrieval.config import RAW_DATA_DIR


def get_mp3_path(track_id: str) -> str:
    """
    Locates the audio file for a given track ID within the FMA dataset structure.
    FMA structure is usually: fma_large/000/000123.mp3
    """

    track_id_str = f"{int(track_id):06d}"

    subfolder = track_id_str[:3]
    fma_path = RAW_DATA_DIR / subfolder / f"{track_id_str}.mp3"

    if fma_path.exists():
        return str(fma_path)

    candidates = list(RAW_DATA_DIR.rglob(f"{track_id_str}.mp3"))
    if candidates:
        return str(candidates[0])

    return None


def render_audio_player(track_id: str):
    """Renders a Streamlit audio player if the file exists."""
    mp3_path = get_mp3_path(track_id)
    if mp3_path:
        st.audio(mp3_path, format="audio/mp3", start_time=0)
    else:
        st.warning(f"Audio file not found for ID {track_id}")
