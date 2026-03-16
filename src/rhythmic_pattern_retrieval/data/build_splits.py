import pandas as pd
import ast
from sklearn.model_selection import train_test_split

from rhythmic_pattern_retrieval.config import METADATA_GENRES_PATH, METADATA_TRACKS_PATH, DATA_DIR

MIN_DURATION = 30
MAX_DURATION = 600


def load_tracks():
    print("Metadata loading...")
    tracks = pd.read_csv(METADATA_TRACKS_PATH, index_col=0, header=[0, 1])
    df = tracks["track"][["duration", "genres_all", "genre_top"]].copy()
    return df


def get_banned_ids():
    genres = pd.read_csv(METADATA_GENRES_PATH, index_col=0)
    banned_names = ["Spoken", "Field Recordings"]
    banned_ids = []
    for name in banned_names:
        try:
            gid = genres[genres["title"].str.contains(
                name, case=False)].index[0]
            banned_ids.append(gid)
        except IndexError:
            pass
    print(f"IDs banned : {banned_ids}")
    return banned_ids


def is_banned(genre_str, banned_ids):
    try:
        g_ids = ast.literal_eval(genre_str)
        return not set(g_ids).isdisjoint(banned_ids)
    except:
        return False


def main():
    df = load_tracks()
    initial_count = len(df)

    # 1. Duration Filter
    mask_duration = (df["duration"] >= MIN_DURATION) & (
        df["duration"] <= MAX_DURATION)
    df = df[mask_duration]
    print(
        f"Duration Filter ({MIN_DURATION}-{MAX_DURATION}s) : {initial_count} -> {len(df)}")

    # 2. Genre Filter
    banned_ids = get_banned_ids()
    mask_banned = df["genres_all"].apply(lambda x: is_banned(x, banned_ids))
    df = df[~mask_banned]
    print(f"Genre Filter (No Spoken/Field) : -> {len(df)} remaining tracks")

    # 3. Split
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=314, shuffle=True)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=314, shuffle=True)

    print("\n--- SPLIT RESULTS ---")
    print(f"TRAIN : {len(train_df)} tracks")
    print(f"VAL   : {len(val_df)} tracks")
    print(f"TEST  : {len(test_df)} tracks")

    # 4. Save
    train_df.index.to_series().to_csv(
        DATA_DIR / "train_ids.csv", index=False, header=False)
    val_df.index.to_series().to_csv(DATA_DIR / "val_ids.csv", index=False, header=False)
    test_df.index.to_series().to_csv(
        DATA_DIR / "test_ids.csv", index=False, header=False)

    test_df.to_csv(DATA_DIR / "test_set_metadata.csv")

    print(f"\n Generated files dans {DATA_DIR}/")


if __name__ == "__main__":
    main()
