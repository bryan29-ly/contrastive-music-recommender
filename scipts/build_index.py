import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from rhythmic_pattern_retrieval.config import PROCESSED_DATA_DIR, MODELS_DIR, DATABASE_DATA_DIR, PROJECT_ROOT
from rhythmic_pattern_retrieval.inference.model_wrapper import GrooveMatcher


def build():
    model_path = MODELS_DIR / "best_model.pth"
    DATABASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load the model
    matcher = GrooveMatcher(model_path)

    files = list(PROCESSED_DATA_DIR.glob("*.pt"))
    print(f"Found {len(files)} files to index.")

    data = []

    # 1. Inference
    print("Computing embeddings...")
    for f in tqdm(files):
        try:
            vec = matcher.get_embeddings(f)
            data.append({
                'track_id': f.stem,
                'path': str(f.relative_to(PROJECT_ROOT)),
                'vector': vec
            })
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

    df = pd.DataFrame(data)
    print(
        f"Embeddings computed. Matrix shape: {len(df)} x {len(df.iloc[0]['vector'])}")

    # 2. t-sne projection
    if len(df) > 0:
        print("Computing PCA (dim reduction)...")
        matrix = np.stack(df['vector'].values)

        pca = PCA(n_components=min(50, matrix.shape[1]))
        matrix_pca = pca.fit_transform(matrix)

        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, init='pca',
                    learning_rate='auto', n_jobs=-1, random_state=42)
        proj = tsne.fit_transform(matrix_pca)

        df['x'] = proj[:, 0]
        df['y'] = proj[:, 1]

        # Saving
        output_path = DATABASE_DATA_DIR / "app_database.pkl"
        df.to_pickle(output_path)
        print(f"Database saved to {output_path} ({len(df)} tracks)")
    else:
        print("Error: No embeddings computed.")


if __name__ == "__main__":
    build()
