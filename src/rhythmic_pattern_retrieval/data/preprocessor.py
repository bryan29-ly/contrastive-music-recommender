import argparse
import torch
import librosa
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

from rhythmic_pattern_retrieval.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_RATE
from rhythmic_pattern_retrieval.utils.audio_utils import compute_mel_spectrogram, resample_audio, get_valid_frames

# os.environ['SSL_CERT_FILE'] = certifi.where()
# os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()


class Preprocessor:
    def __init__(self, limit=None, n_jobs=None):
        self.limit = limit
        self.n_jobs = n_jobs if n_jobs else max(
            1, multiprocessing.cpu_count() - 1)

    def preprocess_file(self, mp3_path):
        save_path = PROCESSED_DATA_DIR / f"{mp3_path.stem}.pt"
        if save_path.exists():
            return None

        try:
            # Loading with Librosa
            wav_np, sr = librosa.load(str(mp3_path), sr=None, mono=True)
            wav = torch.from_numpy(wav_np).float()

            # Resampling
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            if sr != SAMPLE_RATE:
                wav = resample_audio(wav, sr, SAMPLE_RATE)

            # Mel spec
            full_mel_spec = compute_mel_spectrogram(wav, SAMPLE_RATE)

            if full_mel_spec.dim() == 2:
                full_mel_spec = full_mel_spec.unsqueeze(0)

            wav_squeezed = wav.squeeze().numpy()

            valid_indices = get_valid_frames(
                wav_squeezed, hop_length=256, threshold_ratio=0.4)

            data_to_save = {
                "mel": full_mel_spec,  # Tensor
                "valid_indices": valid_indices  # Numpy array
            }

            torch.save(data_to_save, save_path)
            return save_path

        except Exception as e:
            print(f"Error processing {mp3_path.name}: {e}")
            return None

    def preprocess_dataset(self):
        print(
            f"Device for processing: CPU (Multi-core with {self.n_jobs} jobs)")
        audio_files = list(RAW_DATA_DIR.glob("**/*.mp3"))
        if self.limit:
            audio_files = audio_files[:self.limit]

        print(f"Starting processing of {len(audio_files)} files...")

        # Parallelisation
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(self.preprocess_file)(mp3_path)
            for mp3_path in tqdm(audio_files, desc="Preprocessing", unit="track")
        )

        # Count the successes
        success_count = sum(1 for r in results if r is not None)
        print(
            f"Done! {success_count}/{len(audio_files)} processed successfully.")
        print(f"Data saved in {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Dataset for Groove Retrieval")

    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of files to process (useful for testing)")

    parser.add_argument("--jobs", type=int, default=None,
                        help="Number of CPU jobs (default: max - 1)")

    args = parser.parse_args()

    processor = Preprocessor(
        limit=args.limit,
        n_jobs=args.jobs,
    )
    processor.preprocess_dataset()
