import torch
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from rhythmic_pattern_retrieval.config import PROCESSED_DATA_DIR, SAMPLE_RATE


def show_bass_drums_spectrogram():
    # On cherche le fichier spécifique ou on en prend un au hasard
    file_path = PROCESSED_DATA_DIR / "135336.pt"  # Ton fichier test

    if not file_path.exists():
        # Fallback si le fichier spécifique n'existe pas
        files = list(PROCESSED_DATA_DIR.glob("*.pt"))
        if not files:
            print(f"❌ Aucun fichier trouvé dans {PROCESSED_DATA_DIR}")
            return
        file_path = files[0]

    print(f"👀 Visualisation de : {file_path.name}")

    # Chargement du tenseur [2, 128, Time]
    spec_tensor = torch.load(file_path)

    # Vérification de la forme
    if spec_tensor.dim() != 3 or spec_tensor.shape[0] != 2:
        print(
            f"⚠️ Attention: Ce fichier n'a pas le format attendu [2, 128, T]. Shape: {spec_tensor.shape}")
        return

    # Séparation des canaux et conversion en Numpy
    # Canal 0 = Drums, Canal 1 = Bass
    drums_spec = spec_tensor[0].numpy()
    bass_spec = spec_tensor[1].numpy()

    # Création de la figure avec 2 sous-graphiques (l'un sous l'autre)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 1. Plot Drums (Haut)
    img_drums = librosa.display.specshow(
        drums_spec,
        sr=SAMPLE_RATE,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        fmax=8000,
        cmap='magma',
        ax=ax[0]
    )
    ax[0].set_title(f"Drums (Canal 0) - {file_path.stem}")
    fig.colorbar(img_drums, ax=ax[0], format='%+2.0f dB')

    # 2. Plot Bass (Bas)
    img_bass = librosa.display.specshow(
        bass_spec,
        sr=SAMPLE_RATE,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        # On peut réduire fmax pour la basse si on veut zoomer sur les graves (ex: 1000)
        fmax=8000,
        cmap='viridis',  # Changement de couleur pour bien distinguer
        ax=ax[1]
    )
    ax[1].set_title(f"Basse (Canal 1) - {file_path.stem}")
    fig.colorbar(img_bass, ax=ax[1], format='%+2.0f dB')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_bass_drums_spectrogram()
