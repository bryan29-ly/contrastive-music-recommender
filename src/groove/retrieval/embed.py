"""Turn a preprocessed library into one embedding vector per track.

Inference reuses the exact preprocessing baked into the training checkpoint
(audio config + global norm stats), so embeddings are consistent with training.
Each segment is encoded whole -- no random crop, no augmentation -- using the
retrieval representation ``h`` (the projection head is dropped after training);
a track's vector is the L2-normalized mean of its segment vectors.
"""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from groove.audio.features import LogMelSpectrogram
from groove.models.encoder import RhythmicEncoder

# Pretrained weights published on the Hugging Face Hub. Used by default so the
# recommendation pipeline runs without training. Set this to your own repo after
# uploading a checkpoint (see the README).
DEFAULT_CHECKPOINT = "bryan29-ly/groove-rhythm-simclr"


def resolve_checkpoint(ref, filename="best.pt"):
    """Resolve a checkpoint reference to a local file path.

    ``ref`` is either a local path (used as-is) or a Hugging Face Hub repo id,
    optionally ``repo_id:filename``, which is downloaded and cached locally. This
    lets the pretrained model be fetched on demand rather than committed to git.
    """
    if Path(ref).exists():
        return str(ref)
    repo_id, _, fname = ref.partition(":")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=fname or filename)


def load_encoder(checkpoint_path, device):
    """Rebuild the encoder + log-mel front-end from a training checkpoint.

    ``checkpoint_path`` may be a local file or a Hugging Face repo id. The log-mel
    normalization stats are read from the checkpoint config, so inference matches
    training exactly. Returns (model, mel, cfg), both modules in eval mode so
    BatchNorm uses its running statistics and the output is deterministic.
    """
    ckpt = torch.load(resolve_checkpoint(checkpoint_path),
                      map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = RhythmicEncoder(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    mel = LogMelSpectrogram(cfg).to(device).eval()
    return model, mel, cfg


@torch.no_grad()
def embed_segments(manifest_df, segments_dir, model, mel, cfg, device,
                   batch_size=256):
    """Encode each manifest segment into one L2-normalized vector.

    Returns (df, seg_emb [n_segments, D] float32), aligned row-for-row, where
    ``df`` is the manifest with a reset index. Segment-level embeddings are what
    same-track recall and alignment/uniformity are measured on.
    """
    segments_dir = Path(segments_dir)
    seg_len = int(cfg.segment_seconds * cfg.sample_rate)
    df = manifest_df.reset_index(drop=True)
    paths = df["segment_path"].tolist()

    seg_emb = []
    for i in range(0, len(paths), batch_size):
        wavs = []
        for rel in paths[i:i + batch_size]:
            data, _ = sf.read(segments_dir / rel, dtype="float32")
            w = torch.from_numpy(data)
            w = (F.pad(w, (0, seg_len - w.shape[0])) if w.shape[0] < seg_len
                 else w[:seg_len])
            wavs.append(w)
        h, _ = model(mel(torch.stack(wavs).to(device)))
        seg_emb.append(F.normalize(h, dim=1).cpu())
    return df, torch.cat(seg_emb).numpy().astype("float32")


def pool_to_tracks(df, seg_emb):
    """Mean-pool segment vectors per track, then L2-normalize the track vector.

    Returns (track_ids, embeddings [n_tracks, D], bpms [n_tracks]), row-aligned.
    """
    groups = df.groupby("track_id", sort=False).indices
    bpm_of = df.groupby("track_id", sort=False)["bpm"].mean()
    track_ids, vectors, bpms = [], [], []
    for tid, idx in groups.items():
        v = seg_emb[idx].mean(axis=0)
        vectors.append(v / (np.linalg.norm(v) + 1e-8))
        track_ids.append(tid)
        bpms.append(float(bpm_of[tid]))
    return track_ids, np.stack(vectors).astype("float32"), np.asarray(bpms, "float32")


def embed_library(manifest_df, segments_dir, model, mel, cfg, device,
                  batch_size=256):
    """Embed every track into one L2-normalized vector (segment mean-pool).

    Returns (track_ids, embeddings [n_tracks, D] float32, bpms [n_tracks]).
    """
    df, seg_emb = embed_segments(
        manifest_df, segments_dir, model, mel, cfg, device, batch_size)
    return pool_to_tracks(df, seg_emb)
