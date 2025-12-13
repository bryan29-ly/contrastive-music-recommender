import streamlit as st
import plotly.express as px
import pandas as pd

from rhythmic_pattern_retrieval.inference.model_wrapper import GrooveMatcher
from rhythmic_pattern_retrieval.streamlit_ui.utils_ui import get_mp3_path
from rhythmic_pattern_retrieval.config import MODELS_DIR, DATABASE_DATA_DIR, DATA_DIR



# Config
st.set_page_config(page_title="Groove MAtcher", layout="wide")

# Loading
@st.cache_resource
def load_engine():
    # 1. Chargement standard du Moteur (Modèle + Embeddings)
    matcher = GrooveMatcher(
        model_path=MODELS_DIR/"best_model.pth",
        db_path=DATABASE_DATA_DIR/"app_database.pkl"
    )
    
    # 2. Injection des Métadonnées (Genres)
    # On définit le chemin vers le fichier csv de FMA
    metadata_path = DATA_DIR / "metadata" / "fma_metadata" / "tracks.csv"
    
    if metadata_path.exists():
        try:
            # FMA a un format spécifique (Multi-header)
            # index_col=0 car l'ID est la première colonne
            # header=[0, 1] car il y a deux lignes d'en-tête (Artist, Album, Track...)
            meta = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
            
            # On ne garde que la colonne 'genre_top' de la section 'track'
            genres = meta[('track', 'genre_top')]
            
            # --- FUSION ---
            # Le CSV FMA utilise des entiers (123) pour l'index
            # Ta base utilise des strings ("000123") ou entiers selon ton preprocessing
            # On convertit temporairement pour matcher
            matcher.database['track_id_int'] = matcher.database['track_id'].astype(int)
            
            # On map les genres sur les IDs
            matcher.database['Genre'] = matcher.database['track_id_int'].map(genres)
            
            # Remplacer les genres manquants
            matcher.database['Genre'] = matcher.database['Genre'].fillna('Unknown')
            
            print("✅ Métadonnées (Genres) fusionnées avec succès !")
            
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement des métadonnées : {e}")
            matcher.database['Genre'] = 'No Metadata'
    else:
        # Fallback si le fichier csv n'est pas là
        matcher.database['Genre'] = 'No Metadata'
        print(f"⚠️ Fichier tracks.csv introuvable ici : {metadata_path}")

    return matcher

matcher = load_engine()

st.title("Groove Matcher for DJs")

# Tabs
tab1, tab2 = st.tabs(["Explore Library", "Find Similar Track"])

with tab1:
    st.header("Visualization of Rhythmic Space")
    st.write("Each point is a song. Neighbors have the same 'Groove' (Bass/Drums).")

    fig = px.scatter(
    matcher.database, 
    x='x', 
    y='y', 
    color='Genre',  # <--- C'EST ICI QUE LA MAGIE OPÈRE
    hover_data=['track_id', 'Genre'],
    title="Rhythmic map (t-SNE)",
    template="plotly_dark",
    opacity=0.7,
    # Optionnel : définir des couleurs sympas
    color_discrete_sequence=px.colors.qualitative.Bold 
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Mix Recomender")

    track_id = st.selectbox("Select a reference track:", matcher.database['track_id'])

    if st.button("Find Matches"):
        ref_row = matcher.database[matcher.database['track_id'] == track_id].iloc[0]
        query_vec = ref_row['vector']

        # Search
        results = matcher.search(query_vec, top_k=6)

        # Display
        st.write(f"### Top 5 matches for {track_id}")
        cols = st.columns(3)
        for idx, (_, row) in enumerate(results.iloc[1:].iterrows()):
            with cols[idx % 3]:
                match_id = row['track_id']
                score = row['score']
                mp3_path = get_mp3_path(match_id)

                with st.container(border=True):
                    st.markdown(f"**Track {match_id}**")
                    st.progress(float(score), text=f"Match: {score:.1%}")
                    
                    if mp3_path:
                        st.audio(mp3_path, format="audio/mp3", start_time=0)
                    else:
                        st.warning("Audio not found (Local)")
