import argparse
import multiprocessing
from pathlib import Path
import pandas as pd

import torch
import librosa
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from rhythmic_pattern_retrieval.config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_RATE
from rhythmic_pattern_retrieval.utils.audio_utils import compute_mel_spectrogram, resample_audio, get_valid_frames


class Preprocessor:
    def __init__(self, limit=None, n_jobs=None):
        self.limit = limit
        self.n_jobs = n_jobs if n_jobs else max(
            1, multiprocessing.cpu_count() - 1)
        self.target_duration = 30.0
        self.hop_length = 256
        self.energy_threshold = 0.4

    def _get_load_params(self, file_path: Path):
        duration = librosa.get_duration(path=file_path)

        if duration <= self.target_duration:
            return 0.0, None

        start = (duration / 2) - (self.target_duration / 2)
        return start, self.target_duration

    def _get_whitelist_ids(self):
        whitelist = set()
        splits = ["train", "val", "test"]
        found_any = False
        print("Loading whitelist from split files...")
        for split in splits:
            file_path = DATA_DIR / f"{split}_ids.csv"
            if file_path.exists():
                ids = pd.read_csv(file_path, header=None)[0].to_list()
                whitelist.update(ids)
                found_any = True
                print(f"   - Loaded {len(ids)} IDs from {split}_ids.csv")
            else:
                print(f" Warning: {file_path} not found.")

        if not found_any:
            raise FileNotFoundError(
                "0 file found ! Run build_splits.py first.")
        print(f"Total Whitelist: {len(whitelist)} unique tracks to process.")
        return whitelist

    def preprocess_file(self, mp3_path: Path):
        save_path = PROCESSED_DATA_DIR / f"{mp3_path.stem}.pt"
        if save_path.exists():
            return None

        try:
            # 1. Determine load parameters
            offset, duration = self._get_load_params(mp3_path)

            # 2. Loading with Librosa only the specific segment
            wav_np, sr = librosa.load(
                str(mp3_path), sr=None, mono=True, offset=offset, duration=duration)
            wav = torch.from_numpy(wav_np).float()

            # 3. Resampling if necessary
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            if sr != SAMPLE_RATE:
                wav = resample_audio(wav, sr, SAMPLE_RATE)

            # 4. Compute Mel spec
            mel_spec = compute_mel_spectrogram(wav, SAMPLE_RATE)
            if mel_spec.dim() == 2:
                mel_spec = mel_spec.unsqueeze(0)

            # 5. Check for valid signal in the crop
            wav_squeezed = wav.squeeze().numpy()
            valid_indices = get_valid_frames(
                wav_squeezed, hop_length=self.hop_length, threshold_ratio=self.energy_threshold)

            # Discard empty crops
            if len(valid_indices) == 0:
                return None

            data_to_save = {
                "mel": mel_spec,  # Tensor
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

        # 1. Load whitelist
        whitelist = self._get_whitelist_ids()

        # 2. Scan all audio files
        print("Scanning raw audio files...")
        all_audio_files = list(RAW_DATA_DIR.glob("**/*.mp3"))

        # 3. Filter
        audio_files = []
        for f in all_audio_files:
            try:
                track_id = int(f.stem)
                if track_id in whitelist:
                    audio_files.append(f)
            except ValueError:
                continue
        print(
            f"Filtered: Keeping {len(audio_files)} files out of {len(all_audio_files)} found on disk.")

        if self.limit:
            audio_files = audio_files[:self.limit]

        print(f"Starting processing of {len(audio_files)} files...")
        print(f"Center crop of {self.target_duration}s.")

        # Parallelisation
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(self.preprocess_file)(mp3_path)
            for mp3_path in tqdm(audio_files, desc="Preprocessing", unit="track")
        )

        # Count the successes
        success_count = sum(1 for r in results if r is not None)
        print(
            f"Done! {success_count}/{len(audio_files)} processed successfully.")
        print(
            f"Completed: {success_count}/{len(audio_files)} tracks saved in {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset Preprocessing")

    parser.add_argument("--limit", type=int, default=None,
                        help="Limit files for testing")

    parser.add_argument("--jobs", type=int, default=None,
                        help="Number of CPU jobs (default: max - 1)")

    args = parser.parse_args()

    processor = Preprocessor(
        limit=args.limit,
        n_jobs=args.jobs,
    )
    processor.preprocess_dataset()
