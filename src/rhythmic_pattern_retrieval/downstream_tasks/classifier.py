import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from rhythmic_pattern_retrieval.config import DATABASE_DATA_DIR, METADATA_PATH

DATABASE_PATH = DATABASE_DATA_DIR / "app_database.pkl"


def load_fma_metadata(metadata_path, subset="small"):
    print("Metadata loading...")
    tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
    keep_cols = [("set", "subset"), ("track", "genre_top")]
    df = tracks[keep_cols].copy()

    df.columns = ["subset", "genre"]

    if subset:
        df = df[df["subset"] == subset]

    df = df.dropna(subset=["genre"])
    print(f"Loading of {metadata_path.name} done !")
    return df


def main():
    print(f"Loading of the vector base {DATABASE_PATH}")
    df_vectors = pd.read_pickle(DATABASE_PATH)
    print(f"{len(df_vectors)} tracks found.")

    df_labels = load_fma_metadata(METADATA_PATH, subset="small")
    print(f"{len(df_labels)} labels found.")

    df_vectors["clean_id"] = df_vectors["track_id"].astype(
        str).str.extract(r"(\d+)")[0]
    df_vectors["clean_id"] = df_vectors["clean_id"].apply(
        lambda x: f"{int(x):06d}")

    df_labels.index = df_labels.index.map(lambda x: f"{int(x):06d}")

    df_final = df_vectors.merge(
        df_labels, left_on="clean_id", right_index=True)

    if len(df_final) == 0:
        print("❌ ERREUR : Aucune correspondance trouvée entre les IDs du PKL et du CSV.")
        print(
            "Vérifie le format des noms de fichiers (ex: '000123.pt' vs 'track_000123.pt').")
        return

    # 4. Préparation X et y
    print("⚙️ Préparation de la régression logistique...")

    # X : On empile les vecteurs (qui sont des arrays numpy dans la colonne)
    X = np.stack(df_final['vector'].values)

    # y : Les genres
    y = df_final['genre'].values

    # Encodage des labels (Rock -> 0, Pop -> 1)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 5. Split & Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 6. Entraînement
    clf = LogisticRegression(max_iter=2000, multi_class='multinomial')
    clf.fit(X_train, y_train)

    # 7. Résultats
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*40)
    print(f"🏆 RÉSULTAT FINAL (ACCURACY) : {acc:.2%}")
    print("="*40)

    print("\n📊 Détail par genre :")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - Accuracy: {acc:.2%}')
    plt.ylabel('Vrai Genre')
    plt.xlabel('Genre Prédit')
    plt.tight_layout()
    plt.savefig('confusion_matrix_result.png')
    print("🖼️ Matrice de confusion sauvegardée sous 'confusion_matrix_result.png'")


if __name__ == "__main__":
    main()
