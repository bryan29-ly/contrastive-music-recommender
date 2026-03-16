import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from rhythmic_pattern_retrieval.config import DATABASE_DATA_DIR, METADATA_TRACKS_PATH

DATABASE_PATH = "/Users/bryan/Downloads/app_database_wo_attention_pool.pkl"


def load_fma_metadata(metadata_path, subset="small"):
    print("Loading metadata...")
    tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
    keep_cols = [("set", "subset"), ("track", "genre_top")]
    df = tracks[keep_cols].copy()

    df.columns = ["subset", "genre"]

    if subset:
        df = df[df["subset"] == subset]

    df = df.dropna(subset=["genre"])
    print(f"Loading of {metadata_path.name} completed!")
    return df


def main():
    print(f"Loading vector database {DATABASE_PATH}")
    df_vectors = pd.read_pickle(DATABASE_PATH)
    print(f"{len(df_vectors)} tracks found.")

    df_labels = load_fma_metadata(METADATA_TRACKS_PATH, subset="small")
    print(f"{len(df_labels)} labels found.")

    df_vectors["clean_id"] = df_vectors["track_id"].astype(
        str).str.extract(r"(\d+)")[0]
    df_vectors["clean_id"] = df_vectors["clean_id"].apply(
        lambda x: f"{int(x):06d}")

    df_labels.index = df_labels.index.map(lambda x: f"{int(x):06d}")

    df_final = df_vectors.merge(
        df_labels, left_on="clean_id", right_index=True)

    if len(df_final) == 0:
        print("ERROR: No match found between PKL and CSV IDs.")
        return

    # 4. Prepare X and y
    print("Preparing logistic regression...")

    # X: Stack the vectors (which are numpy arrays in the column)
    X = np.stack(df_final['vector'].values)

    # y: The genres
    y = df_final['genre'].values

    # Label encoding (Rock -> 0, Pop -> 1)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 5. Split & Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 6. Training
    clf = LogisticRegression(max_iter=2000, multi_class='multinomial')
    clf.fit(X_train, y_train)

    # 7. Results
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*40)
    print(f"FINAL RESULT (ACCURACY): {acc:.2%}")
    print("="*40)

    print("\nDetails by genre:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - Accuracy: {acc:.2%}')
    plt.ylabel('True Genre')
    plt.xlabel('Predicted Genre')
    plt.tight_layout()
    plt.savefig('confusion_matrix_result.png')
    print("Confusion matrix saved as 'confusion_matrix_result.png'")


if __name__ == "__main__":
    main()
