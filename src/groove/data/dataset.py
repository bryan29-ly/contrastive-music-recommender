import math
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
                 hop_length, cross_segment_prob, stretch_min=1.0, limit=None):
        self.segments_dir = Path(segments_dir)
        self.cross_segment_prob = cross_segment_prob
        # Read the context the slowest stretch factor needs: the augmentation
        # crops back to crop_frames, so a slow-down never runs out of frames.
        # Clamped at 1.0 -- a range that only speeds up must still yield a full
        # crop on the calls where no stretch is drawn.
        context_frames = math.ceil(crop_frames / min(1.0, stretch_min))
        self.crop_samples = (context_frames - 1) * hop_length

        df = pd.read_csv(manifest_path, dtype={"track_id": str})
        df = df[df["split"] == split].reset_index(drop=True)

        # One dataset item per *track* (instance discrimination): a track is
        # drawn once per epoch and its two views are the only positive pair, so
        # two segments of the same track are never sampled as separate anchors
        # in a batch -> no same-track false negatives in the NT-Xent loss.
        self.by_track = df.groupby("track_id")["segment_path"].apply(list).to_dict()
        self.track_ids = list(self.by_track.keys())
        if limit is not None:  # debug mode: keep a small subset
            self.track_ids = self.track_ids[:limit]

        n_seg = sum(len(self.by_track[t]) for t in self.track_ids)
        print(f"[{split}] {len(self.track_ids)} tracks / {n_seg} segments.")

    def __len__(self):
        return len(self.track_ids)

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
        segments = self.by_track[self.track_ids[index]]
        anchor = random.choice(segments)

        view1 = self._load_crop(anchor)
        if len(segments) > 1 and random.random() < self.cross_segment_prob:
            other = random.choice([s for s in segments if s != anchor])
            view2 = self._load_crop(other)
        else:
            view2 = self._load_crop(anchor)  # independent crop of same segment
        return view1, view2
