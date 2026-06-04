import random
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset


class ContrastivePairDataset(Dataset):
    """Yields two augmentation-ready waveform crops forming a positive pair.

    A positive pair is built from the same track. With probability
    ``cross_segment_prob`` (and if the track has more than one segment) the two
    crops come from *different* segments of the track (the global signature);
    otherwise they are two independent random crops of the same segment. Mel
    spectrograms and augmentations are applied later, on GPU.
    """

    def __init__(self, manifest_path, segments_dir, split, crop_frames,
                 hop_length, cross_segment_prob, limit=None):
        self.segments_dir = Path(segments_dir)
        self.cross_segment_prob = cross_segment_prob
        # crop length in samples chosen so the mel yields exactly crop_frames.
        self.crop_samples = (crop_frames - 1) * hop_length

        df = pd.read_csv(manifest_path, dtype={"track_id": str})
        df = df[df["split"] == split].reset_index(drop=True)

        # One dataset item per segment; the segment is the pair's anchor.
        self.items = df["segment_path"].tolist()
        if limit is not None:  # debug mode: keep a small subset
            self.items = self.items[:limit]
        # Group segment paths by track to find cross-segment partners.
        self.by_track = df.groupby("track_id")["segment_path"].apply(list).to_dict()
        self.track_of = dict(zip(df["segment_path"], df["track_id"]))

        print(f"[{split}] {len(self.items)} segments / "
              f"{len(self.by_track)} tracks.")

    def __len__(self):
        return len(self.items)

    def _load_crop(self, rel_path):
        """Read a random crop_samples window from a FLAC segment (mono float32)."""
        path = self.segments_dir / rel_path
        with sf.SoundFile(path) as f:
            total = len(f)
            if total <= self.crop_samples:
                data = f.read(dtype="float32")
            else:
                start = random.randint(0, total - self.crop_samples)
                f.seek(start)
                data = f.read(self.crop_samples, dtype="float32")
        wav = torch.from_numpy(data)
        if wav.shape[0] < self.crop_samples:  # pad short segments
            wav = torch.nn.functional.pad(wav, (0, self.crop_samples - wav.shape[0]))
        return wav

    def __getitem__(self, index):
        anchor = self.items[index]
        siblings = self.by_track[self.track_of[anchor]]

        view1 = self._load_crop(anchor)
        if len(siblings) > 1 and random.random() < self.cross_segment_prob:
            other = random.choice([s for s in siblings if s != anchor])
            view2 = self._load_crop(other)
        else:
            view2 = self._load_crop(anchor)  # independent crop of same segment
        return view1, view2
