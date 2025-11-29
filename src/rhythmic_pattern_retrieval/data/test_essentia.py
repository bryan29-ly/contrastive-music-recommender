import os
import numpy as np
from pathlib import Path
from essentia import Pool, array
from essentia.standard import MonoLoader, TensorflowPredict, TensorflowInputMusiCNN, FrameGenerator

from rhythmic_pattern_retrieval.config import PROJECT_ROOT

# --- CONFIGURATION ---
AUDIO_FILE = Path(PROJECT_ROOT / "data/raw/fma_small/135/135337.mp3")
MODEL_PATH = PROJECT_ROOT / "models_output/emomusic-msd-musicnn-2.pb"


def test_essentia():
    print(f"Testing Essentia on {AUDIO_FILE}...")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERREUR: Modèle introuvable à {MODEL_PATH}")
        return

    try:
        audio_path_str = str(AUDIO_FILE)
        model_path_str = str(MODEL_PATH)

        # 1. Chargement Audio
        loader = MonoLoader(filename=audio_path_str,
                            sampleRate=16000, resampleQuality=4)
        audio = loader()
        print("✅ Audio chargé.")

        # 2. Préparation
        input_layer_name = "model/Placeholder"
        output_layer_name = "model/Identity"

        # On instancie le modèle UNE FOIS
        model = TensorflowPredict(
            graphFilename=model_path_str,
            inputs=[input_layer_name],
            outputs=[output_layer_name]
        )

        preproc = TensorflowInputMusiCNN()
        features_list = []

        # 3. Extraction des Patches
        for frame in FrameGenerator(audio, frameSize=512, hopSize=256, startFromZero=True):
            patch = preproc(frame)

            if patch.size > 0:
                features_list.append(patch)

        if not features_list:
            print("❌ Aucun patch généré.")
            return

        # 4. Conversion en Tenseur 4D (L'étape CRITIQUE)

        # A. On crée un tableau Numpy 3D : [N_Patches, 187, 96]
        # np.array sur une liste de matrices crée proprement la 3ème dimension
        features_3d = np.array(features_list, dtype=np.float32)

        # B. On ajoute la 4ème dimension (Channel=1) pour faire plaisir à Tensorflow
        # Résultat : [N_Patches, 187, 96, 1]
        features_4d = np.expand_dims(features_3d, axis=-1)

        print(f"✅ Features formatées pour CNN. Shape: {features_4d.shape}")

        # 5. Prédiction par Batch (Plus rapide et stable)
        # Au lieu de boucler, on envoie tout le paquet au Pool
        pool = Pool()

        # essentia.array() convertit le numpy array en format C++ compatible
        pool.set(input_layer_name, array(features_4d))

        # Le modèle traite tout d'un coup
        pool_out = model(pool)
        predictions = pool_out[output_layer_name]  # Shape: [N_Patches, 2]

        # 6. Résultats
        avg_prediction = np.mean(predictions, axis=0)

        valence = float(avg_prediction[0])
        arousal = float(avg_prediction[1])

        print(f"✅ SUCCÈS !")
        print(f"   - Valence : {valence:.2f} / 9")
        print(f"   - Arousal : {arousal:.2f} / 9")

        norm_energy = (arousal - 1) / 8.0
        norm_energy = max(0.0, min(1.0, norm_energy))
        print(f"   -> Label normalisé : {norm_energy:.4f}")

    except Exception as e:
        print(f"❌ Erreur Globale : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_essentia()
