from groove.audio.features import LogMelSpectrogram
from groove.data.dataset import ContrastivePairDataset
from groove.data.transforms import RhythmSafeAugment
from groove.models.encoder import RhythmicEncoder
from groove.models.losses import NTXentLoss
from groove.training.metrics import representation_metrics
from groove.utils.device import get_device
from groove.utils.seed import seed_everything
import csv
import json
import logging
import math
import sys

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from omegaconf import OmegaConf
import torch
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


logger = logging.getLogger("groove")


# --- setup helpers ----------------------------------------------------------

def _setup_run_dir(cfg, name):
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(cfg.out_dir) / (f"{stamp}_{name}" if name else stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _setup_logger(run_dir):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    # stdout so epoch summaries stay separate from tqdm bars (stderr).
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)


def _seed_worker(_):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    import random
    random.seed(seed)


def _build_loaders(cfg, debug):
    common = dict(
        manifest_path=Path(cfg.segments_dir) / "manifest.csv",
        segments_dir=cfg.segments_dir,
        crop_frames=cfg.crop_frames,
        hop_length=cfg.hop_length,
        cross_segment_prob=cfg.cross_segment_prob,
        limit=200 if debug else None,
    )
    train_ds = ContrastivePairDataset(split="train", **common)
    val_ds = ContrastivePairDataset(split="val", **common)
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    def make(ds, shuffle):
        return DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=_seed_worker, generator=g)

    return make(train_ds, True), make(val_ds, False)


def _cosine_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss, cfg):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "config": OmegaConf.to_container(cfg),
    }, path)


def _save_curves(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc, ax_au) = plt.subplots(1, 3, figsize=(16, 5))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax_loss.set(xlabel="epoch", ylabel="NT-Xent loss", title="Contrastive loss")

    # In-batch retrieval accuracy: the most readable proxy for the actual task.
    ax_acc.plot(epochs, [h["val_acc1"] for h in history], label="top-1")
    ax_acc.plot(epochs, [h["val_acc5"] for h in history], label="top-5")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="In-batch retrieval (val)")

    # Wang & Isola (2020) proxies; both go down as the space improves.
    ax_au.plot(epochs, [h["val_align"] for h in history], label="alignment")
    ax_au.plot(epochs, [h["val_uniform"] for h in history], label="uniformity")
    ax_au.set(xlabel="epoch", title="Alignment / uniformity (val)")

    for ax in (ax_loss, ax_acc, ax_au):
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- training ---------------------------------------------------------------

def _forward_pair(v1, v2, mel, augment, model, criterion, device, amp_ctx):
    v1 = v1.to(device, non_blocking=True)
    v2 = v2.to(device, non_blocking=True)
    with amp_ctx():
        m1 = augment(mel(v1))
        m2 = augment(mel(v2))
        _, z1 = model(m1)
        _, z2 = model(m2)
        loss = criterion(z1, z2)
    return loss, z1, z2


def train(cfg, name=None, debug=False):
    seed_everything(cfg.seed, deterministic=debug)
    device = get_device()
    run_dir = _setup_run_dir(cfg, name)
    _setup_logger(run_dir)

    # Load global standardization stats (computed by compute_norm_stats.py).
    stats_path = Path(cfg.segments_dir) / "norm_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        cfg.norm_mean, cfg.norm_std = stats["mean"], stats["std"]
        logger.info(
            f"norm stats: mean={stats['mean']:.3f} std={stats['std']:.3f}")
    else:
        logger.info(
            "no norm_stats.json -> raw log-mel (run compute_norm_stats.py)")

    OmegaConf.save(cfg, run_dir / "config.yaml")
    logger.info(f"run dir: {run_dir}")

    use_amp = (device.type == "cuda") and cfg.amp
    scaler = GradScaler(enabled=use_amp)
    # autocast only on CUDA; a no-op context elsewhere (MPS/CPU).

    def amp_ctx():
        return autocast("cuda") if use_amp else nullcontext()

    train_loader, val_loader = _build_loaders(cfg, debug)
    logger.info(f"device={device.type} amp={use_amp} batch={cfg.batch_size} "
                f"accum={cfg.accum_steps} | train_batches={len(train_loader)} "
                f"val_batches={len(val_loader)}")

    mel = LogMelSpectrogram(cfg).to(device)
    augment = RhythmSafeAugment(cfg).to(device)
    model = RhythmicEncoder(cfg).to(device)
    criterion = NTXentLoss(cfg.temperature).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    accum = max(1, cfg.accum_steps)
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = _cosine_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps)

    history, history_path = [], run_dir / "history.csv"
    # Best model is tracked on an EMA-smoothed val loss: the contrastive val
    # loss is noisy, so the raw minimum often lands on a variance dip rather than
    # a genuinely better epoch. last.pt (final cosine epoch) is the deliverable;
    # best.pt is a safety net. Early stopping is off by default (trust the
    # schedule's low-LR annealing); enable it via cfg.early_stopping.
    best_val, ema_val, patience = float("inf"), None, 0
    early_stopping = bool(cfg.get("early_stopping", False))

    for epoch in range(1, cfg.epochs + 1):
        # --- train ---
        model.train()
        optimizer.zero_grad()
        train_loss, n = 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs} [train]",
                   leave=False)
        for i, (v1, v2) in enumerate(bar):
            loss, _, _ = _forward_pair(v1, v2, mel, augment, model, criterion,
                                       device, amp_ctx)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            train_loss += loss.item()
            n += 1
            bar.set_postfix(loss=f"{loss.item():.3f}")
        train_loss /= max(1, n)

        # --- validation (+ cheap in-batch retrieval proxies) ---
        model.eval()
        val_loss, n = 0.0, 0
        agg = {"align": 0.0, "uniform": 0.0, "acc1": 0.0,
               "acc5": 0.0, "emb_std": 0.0}
        with torch.no_grad():
            for v1, v2 in tqdm(val_loader, desc=f"epoch {epoch}/{cfg.epochs} [val]",
                               leave=False):
                loss, z1, z2 = _forward_pair(v1, v2, mel, augment, model,
                                             criterion, device, amp_ctx)
                metrics = representation_metrics(z1, z2)
                val_loss += loss.item()
                for key in agg:
                    agg[key] += metrics[key]
                n += 1
        val_loss /= max(1, n)
        for key in agg:
            agg[key] /= max(1, n)

        lr = optimizer.param_groups[0]["lr"]
        ema_val = val_loss if ema_val is None else 0.7 * ema_val + 0.3 * val_loss
        logger.info(f"epoch {epoch}/{cfg.epochs} | train {train_loss:.4f} | "
                    f"val {val_loss:.4f} | acc1 {agg['acc1']:.3f} | "
                    f"align {agg['align']:.3f} | uniform {agg['uniform']:.3f} | "
                    f"std {agg['emb_std']:.3f} | lr {lr:.2e}")

        # --- log + checkpoints ---
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss,
                        **{f"val_{k}": v for k, v in agg.items()}, "lr": lr})
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        _save_curves(history, run_dir / "curves.png")
        _save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler,
                         epoch, val_loss, cfg)

        if ema_val < best_val:
            best_val, patience = ema_val, 0
            _save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler,
                             epoch, val_loss, cfg)
            logger.info("  new best model saved (smoothed val loss)")
        else:
            patience += 1
            if early_stopping and patience >= cfg.patience:
                logger.info(f"early stopping at epoch {epoch}")
                break

    logger.info(
        f"done. best smoothed val loss = {best_val:.4f} | artifacts in {run_dir}")
    return run_dir
