import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from essentia.standard import MonoLoader, TensorflowPredict

from rhythmic_pattern_retrieval.config import RAW_DATA_DIR, DATA_DIR, MODELS_OUTPUT_DIR, PROCESSED_DATA_DIR

MODEL_PATH = MODELS_OUTPUT_DIR/"models_output/emomusic-msd-musicnn-2.pb"


def get_audio_files(directory):
    return list(Path(directory).glob("**/*.mp3"))


def main():
    print("--- Starting of metadata extraction (Mood/Energy) ---")

    # Check the model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Essentia model not founded here {MODEL_PATH}. Download it!")

    # Initialization of Essentia
    loader = MonoLoader(sampleRate=16000, resampleQuality=4)
    model = TensorflowPredict(
        graphFilename=MODEL_PATH, output="model.Identity")

    # Get the files
    mp3_files = get_audio_files(RAW_DATA_DIR)

    # Check if corresponding pt exists
    existing_pt_stems = {f.stem for f in PROCESSED_DATA_DIR.glob("*.pt")}

    records = []

    for mp3_path in tqdm(mp3_files, desc="Analyse Essentia"):
        track_id = mp3_path.stem
        if track_id not in existing_pt_stems:
            continue
        try:
            loader.configure(filename=str(mp3_path))
            audio = loader()

            # Prediction shape (1, 2) -> [[valence_val, arousal_val]]
            predictions = model(audio)
            valence = float(predictions[0, 0])  # Positivity (1 to 9)
            arousal = float(predictions[0, 1])  # Energy (1 to 9)
            norm_valence = (valence - 1) / 8.0
            norm_arousal = (arousal - 1) / 8.0

            records.append({
                "track_id": track_id,
                "filename": f"{track_id}.pt",
                "valence_raw": valence,
                "arousal_raw": arousal,
                "mood_valence": norm_valence,  # for the Loss
                "energy_arousal": norm_arousal  # for the Loss
            })

        except Exception as e:
            print(f"Erreur sur {mp3_path.name}: {e}")

    # Save
    df = pd.DataFrame(records)
    df.to_csv(DATA_DIR, index=False)

    print(f"Done ! {len(df)} tracks analyzed.")
    print(f"Files saved : {DATA_DIR}")
    print(df.head())


if __name__ == "__main__":
    main()
