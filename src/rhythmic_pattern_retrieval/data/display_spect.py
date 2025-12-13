import torch
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from rhythmic_pattern_retrieval.config import PROCESSED_DATA_DIR, SAMPLE_RATE

# IMPORTANT : Doit correspondre à la valeur utilisée dans le préprocesseur
HOP_LENGTH = 256


def show_mel_spectrogram(track_name):
    """
    Affiche le Mel Spectrogramme et les zones de 'Valid Indices' détectées.
    """
    # 1. Recherche du fichier
    file_path = PROCESSED_DATA_DIR / track_name

    # Si le fichier exact n'existe pas, on cherche s'il a été renommé ou on prend le premier dispo
    if not file_path.exists():
        print(
            f"⚠️ Fichier {track_name} introuvable. Recherche d'un fichier .pt disponible...")
        files = list(PROCESSED_DATA_DIR.glob("*.pt"))
        if not files:
            print(f"❌ Aucun fichier .pt trouvé dans {PROCESSED_DATA_DIR}")
            return
        file_path = files[0]

    print(f"👀 Visualisation de : {file_path.name}")

    # 2. Chargement (CORRECTION ICI)
    # weights_only=False est nécessaire car on charge des numpy arrays
    data = torch.load(file_path, weights_only=False)

    # Gestion de la compatibilité (Dictionnaire vs Ancien Tenseur)
    if isinstance(data, dict):
        mel_tensor = data["mel"]             # Shape attendue: [1, 128, T]
        valid_indices = data["valid_indices"]  # Numpy array
    else:
        # Cas legacy
        mel_tensor = data
        valid_indices = []

    # 3. Préparation pour l'affichage (Conversion Tensor -> Numpy 2D)
    if mel_tensor.dim() == 3:
        mel_spec = mel_tensor.squeeze(0).numpy()
    elif mel_tensor.dim() == 2:
        mel_spec = mel_tensor.numpy()
    else:
        print(f"❌ Format de tenseur inconnu : {mel_tensor.shape}")
        return

    # 4. Affichage
    plt.figure(figsize=(14, 6))

    # A. Le Spectrogramme
    librosa.display.specshow(
        mel_spec,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis='time',
        y_axis='mel',
        fmax=8000,
        cmap='magma'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel Spectrogram - {file_path.stem}")

    # B. Visualisation des 'Valid Indices'
    if len(valid_indices) > 0:
        times = valid_indices * HOP_LENGTH / SAMPLE_RATE
        plt.vlines(times, ymin=0, ymax=10, color='cyan',
                   alpha=0.5, linewidth=1, label='Valid Anchor')
        plt.legend(loc='upper right')
        print(f"✅ {len(valid_indices)} points d'intérêt (Groove) détectés.")
    else:
        print("⚠️ Aucun index valide trouvé (Silence ou seuil trop haut).")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test avec un fichier qui existe chez toi (ex: le premier trouvé)
    show_mel_spectrogram("109537.pt")
