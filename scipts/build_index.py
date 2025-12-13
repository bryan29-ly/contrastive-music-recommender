import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE

from rhythmic_pattern_retrieval.config import PROCESSED_DATA_DIR, MODELS_DIR, DATABASE_DATA_DIR, PROJECT_ROOT
from rhythmic_pattern_retrieval.inference.model_wrapper import GrooveMatcher

def build():
    model_path = MODELS_DIR / "best_model.pth"
    matcher = GrooveMatcher(model_path)

    files = list(PROCESSED_DATA_DIR.glob("*.pt"))
    print(f"Found {len(files)} files to index.")

    data = []

    # Inference
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
            print(f"Error {f}: {e}")

    df = pd.DataFrame(data)

    # t-SNE compute
    print("Computing t-SNE projection...")
    matrix = np.stack(df['vector'].values)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    proj = tsne.fit_transform(matrix)

    df['x'] = proj[:, 0]
    df['y'] = proj[:, 1]

    # Saving
    output_path = DATABASE_DATA_DIR / "app_database.pkl"
    df.to_pickle(output_path)
    print(f"Database saved to {output_path} ({len(df)} tracks)")

if __name__ == "__main__":
    build()