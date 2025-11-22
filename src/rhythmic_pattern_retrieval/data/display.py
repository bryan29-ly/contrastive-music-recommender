import torch
import matplotlib.pyplot as plt
import librosa.display

class SpectrogramDisplay:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def show(self, spec_tensor, title=None):
        spec_np = spec_tensor.squeeze().numpy()
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(
            spec_np,
            sr=self.sample_rate,
            hop_length=512,
            x_axis='time',
            y_axis='mel',
            fmax=8000,
            cmap='magma',
        )
        plt.title(title or "Spectrogram")
        plt.tight_layout()
        plt.show()
