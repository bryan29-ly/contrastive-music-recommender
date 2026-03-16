import streamlit as st
import plotly.express as px
import pandas as pd

from rhythmic_pattern_retrieval.inference.model_wrapper import GrooveMatcher
from rhythmic_pattern_retrieval.streamlit_ui.utils_ui import get_mp3_path, render_audio_player
from rhythmic_pattern_retrieval.config import MODELS_DIR, DATABASE_DATA_DIR, DATA_DIR

# --- Configuration ---
st.set_page_config(page_title="Groove Matcher", layout="wide", page_icon="🎵")

# --- Engine Loading ---


@st.cache_resource
def load_engine():
    """
    Loads the model, vector database, and FMA metadata (Genres & Splits).
    """
    # 1. Load Inference Engine
    matcher = GrooveMatcher(
        model_path=MODELS_DIR / "best_model.pth",
        db_path="/Users/bryan/Documents/Work 2025/PROJETS/rhythmic-pattern-retrieval/data/database/app_database.pkl"
    )

    # 2. Load Metadata (FMA tracks.csv)
    metadata_path = DATA_DIR / "metadata" / "fma_metadata" / "tracks.csv"

    if metadata_path.exists():
        try:
            # FMA CSV has a multi-index header (rows 0, 1) and ID as index (col 0)
            meta = pd.read_csv(metadata_path, index_col=0, header=[0, 1])

            # Extract relevant columns: Genre and Split (training/validation/test)
            genres = meta[('track', 'genre_top')]
            splits = meta[('set', 'split')]

            # Prepare database for merging
            # Ensure ID match (FMA uses Int, our DB might use padded Strings)
            matcher.database['track_id_int'] = matcher.database['track_id'].astype(
                int)

            # Map Metadata
            matcher.database['Genre'] = matcher.database['track_id_int'].map(
                genres).fillna('Unknown')
            matcher.database['Split'] = matcher.database['track_id_int'].map(
                splits).fillna('Unknown')

            print("Metadata merged successfully.")

        except Exception as e:
            st.error(f"Error loading metadata: {e}")
            matcher.database['Genre'] = 'Unknown'
            matcher.database['Split'] = 'Unknown'
    else:
        print(f"Metadata file not found at {metadata_path}")
        matcher.database['Genre'] = 'Unknown'
        matcher.database['Split'] = 'Unknown'

    return matcher


# Initialize
matcher = load_engine()

# --- Sidebar Controls ---
st.sidebar.title("Controls")
selected_split = st.sidebar.selectbox(
    "Filter by Dataset Split:",
    ["All", "training", "validation", "test"],
    index=0
)

# Filter the dataframe based on selection
if selected_split != "All":
    filtered_db = matcher.database[matcher.database['Split'] == selected_split]
else:
    filtered_db = matcher.database

st.sidebar.markdown(f"**Tracks available:** {len(filtered_db)}")

# --- Main UI ---
st.title("🎹 Groove Matcher")
st.markdown("Rhythmic similarity search based on Contrastive Learning.")

tab1, tab2 = st.tabs(["Find Similar Tracks", "Explore Space"])

# --- TAB 1: Recommendation ---
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Select Reference")
        # Dropdown to select a track
        track_id = st.selectbox(
            "Search by ID:",
            filtered_db['track_id'],
            help="Select a track to find rhythmic neighbors."
        )

        # Get details of selected track
        ref_row = matcher.database[matcher.database['track_id']
                                   == track_id].iloc[0]

        with st.container(border=True):
            st.markdown(f"**Current Track:** `{track_id}`")
            st.markdown(f"**Genre:** {ref_row['Genre']}")
            st.markdown(f"**Split:** {ref_row['Split']}")
            st.markdown("---")
            st.markdown("**Listen to Reference:**")
            render_audio_player(track_id)

    with col2:
        st.subheader("2. Recommendations")

        if st.button("Find Matches", type="primary"):
            # Perform search
            query_vec = ref_row['vector']
            # Top 7 (1 ref + 6 matches)
            results = matcher.search(query_vec, top_k=7)

            # Grid Layout for results
            res_cols = st.columns(3)

            # Skip the first result (it's the reference track itself)
            for idx, (_, row) in enumerate(results.iloc[1:].iterrows()):
                match_id = row['track_id']
                score = row['score']
                match_genre = row['Genre']
                match_split = row['Split']

                with res_cols[idx % 3]:
                    with st.container(border=True):
                        st.markdown(f"### #{idx+1}")
                        st.caption(f"ID: {match_id}")

                        # Similarity Bar
                        st.progress(
                            float(score), text=f"Similarity: {score:.1%}")

                        # Metadata tags
                        st.markdown(
                            f"**{match_genre}** | *{match_split}*")

                        # Audio Player
                        render_audio_player(match_id)

# --- TAB 2: Visualization ---
with tab2:
    st.subheader("Latent Space Projection (t-SNE)")

    # Toggle coloring mode
    color_mode = st.radio("Color by:", ["Genre", "Split"], horizontal=True)

    fig = px.scatter(
        filtered_db,  # Uses the filtered database (based on sidebar)
        x='x',
        y='y',
        color=color_mode,
        hover_data=['track_id', 'Genre', 'Split'],
        template="plotly_dark",
        opacity=0.7,
        height=700,
        title=f"Rhythmic Map ({selected_split} set)"
    )
    st.plotly_chart(fig, use_container_width=True)
