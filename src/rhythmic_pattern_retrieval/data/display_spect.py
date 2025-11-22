import torch
from rhythmic_pattern_retrieval.config import PROCESSED_DATA_DIR, SAMPLE_RATE
from rhythmic_pattern_retrieval.data.display import SpectrogramDisplay

def show_random_spectrogram():
    import matplotlib.pyplot as plt
    import librosa.display
    files = list(PROCESSED_DATA_DIR.glob("*.pt"))
    if not files:
        print(f"No files found in {PROCESSED_DATA_DIR}")
        return
    file_path_1 = PROCESSED_DATA_DIR/"135336_old.pt"
    file_path_2 = PROCESSED_DATA_DIR/"135336.pt"
    print(f"Visualizing: {file_path_1.name}")
    print(f"Visualizing: {file_path_2.name}")
    spec_tensor_1 = torch.load(file_path_1)
    spec_tensor_2 = torch.load(file_path_2)
    spec_np_1 = spec_tensor_1.squeeze().numpy()
    spec_np_2 = spec_tensor_2.squeeze().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    librosa.display.specshow(
        spec_np_1,
        sr=SAMPLE_RATE,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        fmax=8000,
        cmap='magma',
        ax=ax[0]
    )
    ax[0].set_title(f"Fichier: {file_path_1.name}")
    librosa.display.specshow(
        spec_np_2,
        sr=SAMPLE_RATE,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        fmax=8000,
        cmap='magma',
        ax=ax[1]
    )
    ax[1].set_title(f"Fichier: {file_path_2.name}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_random_spectrogram()
