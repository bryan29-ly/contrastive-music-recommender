import argparse
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import sys

from rhythmic_pattern_retrieval.utils.reproducibility import seed_everything
from rhythmic_pattern_retrieval.models.loss import NTXentLoss
from rhythmic_pattern_retrieval.models.encoder import RhythmicEncoder
from rhythmic_pattern_retrieval.pipeline.data_pipeline import create_contrastive_dataloader
from rhythmic_pattern_retrieval.utils.device import get_device
from rhythmic_pattern_retrieval.data.dataset import SpectrogramDataset
from rhythmic_pattern_retrieval.train.metrics import compute_alignment_uniformity
from rhythmic_pattern_retrieval.config import PROCESSED_DATA_DIR, MODELS_DIR, DATA_DIR
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_args():
    parser = argparse.ArgumentParser(
        description="Train Groove Contrastive Model")
    # Parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (Mac: 32-64, RunPod: 256-512)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Num workers dataloader")
    parser.add_argument("--patience", type=int, default=75, help="Patience")

    # Training parameter
    parser.add_argument("--epochs", type=int, default=5,
                        help="Nombre d'epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--crop_size", type=int, default=512,
                        help="Taille du crop temporel")

    return parser.parse_args()


def save_plots(history_df):
    """Save loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1 : Losses
    ax1.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    ax1.plot(history_df['epoch'], history_df['val_loss'],
             label='Validation Loss', linestyle='--')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('NT-Xent Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot 2 : Metrics
    ax2.set_xlabel('Epochs')
    ax2.set_title('Alignment & Uniformity (Validation)')
    ax2.grid(True)

    color = 'tab:blue'
    ax2.set_ylabel('Alignment', color=color)
    ax2.plot(history_df['epoch'], history_df['align'],
             color=color, label='Alignment')
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax2.twinx()  # Deuxième axe Y
    color = 'tab:orange'
    ax3.set_ylabel('Uniformity', color=color)
    ax3.plot(history_df['epoch'], history_df['unif'],
             color=color, linestyle='--', label='Uniformity')
    ax3.tick_params(axis='y', labelcolor=color)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "training_curves.png")
    plt.close()


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, args, filename="checkpoint.pth"):
    "Save the state of the model"
    save_path = MODELS_DIR / filename
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'args': vars(args),
        'loss': loss
    }, save_path)


def get_cosine_schedule_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    args = get_args()
    device = get_device()

    print("\nConfiguration:")
    print(f"   - Device: {device.type.upper()}")
    print(f"   - Batch Size: {args.batch_size}")
    print(f"   - Epochs: {args.epochs}")

    # --- CONFIGURATION HYBRIDE ---
    use_amp = (device.type == 'cuda')

    # Le Scaler n'est créé que si on utilise AMP
    scaler = GradScaler('cuda') if use_amp else None

    if use_amp:
        print("Mixed Precision (AMP) Enabled")
        torch.backends.cudnn.benchmark = True
    else:
        print("Running in standard FP32 mode (Safer for Mac/CPU)")

    # Data pipeline
    print("Loading data...")

    # split Train / Validatioin
    train_ids = pd.read_csv(DATA_DIR / "train_ids.csv",
                            header=None)[0].tolist()
    val_ids = pd.read_csv(DATA_DIR / "val_ids.csv", header=None)[0].tolist()

    train_dataset = SpectrogramDataset(
        root_dir=PROCESSED_DATA_DIR, ids_list=train_ids, crop_size=args.crop_size)
    val_dataset = SpectrogramDataset(
        root_dir=PROCESSED_DATA_DIR, ids_list=val_ids, crop_size=args.crop_size)

    print(f"    - Train samples : {len(train_dataset)}")
    print(f"    - Val samples : {len(val_dataset)}")

    # Loaders with pipeline
    train_loader, augment = create_contrastive_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        shuffle=True,
        num_workers=args.workers
    )

    val_loader, _ = create_contrastive_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        shuffle=False,
        num_workers=args.workers
    )

    # Model & Optimisation
    model = RhythmicEncoder(projection_dim=128).to(device)
    augment = augment.to(device)  # on GPU

    criterion = NTXentLoss(temperature=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Steps for scheduler
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * 0.1)  # 10% of the training for the warmup

    # Scheduler
    scheduler = get_cosine_schedule_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Monitoring
    history = []
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    print("\n Starting training...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0

        # Get the current lr
        current_lr = optimizer.param_groups[0]['lr']

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [LR={current_lr:.2e}]")

        for view1_raw, view2_raw, _ in progress_bar:
            view1_raw = view1_raw.to(device, non_blocking=True)
            view2_raw = view2_raw.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed Precision (AMP)
            with autocast(device_type=device.type, enabled=use_amp):
                view1 = augment(view1_raw)
                view2 = augment(view2_raw)

                # Forward
                _, z1 = model(view1)
                _, z2 = model(view2)

                # Loss
                loss = criterion(z1, z2)

            # Optimized Backward
            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                scheduler.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        total_val_loss = 0
        total_align = 0
        total_unif = 0

        with torch.no_grad():
            for view1_raw, view2_raw, _ in val_loader:
                view1_raw = view1_raw.to(device)
                view2_raw = view2_raw.to(device)

                with autocast(device_type=device.type, enabled=use_amp):
                    view1 = augment(view1_raw)
                    view2 = augment(view2_raw)

                    _, z1 = model(view1)
                    _, z2 = model(view2)
                    loss = criterion(z1, z2)
                    align, unif = compute_alignment_uniformity(z1, z2)

                total_val_loss += loss.item()
                total_align += align.item()
                total_unif += unif.item()

        # Stats
        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        avg_align = total_align / len(val_loader)
        avg_unif = total_unif / len(val_loader)

        # Logging & Save
        print(
            f"   Stats: Train Loss = {avg_train:.4f} | Val Loss = {avg_val:.4f}")
        print(f"   Metrics: Align = {avg_align:.4f} | Unif = {avg_unif:.4f}")

        # Save in the history
        history.append(
            {'epoch': epoch + 1, 'train_loss': avg_train, 'val_loss': avg_val, 'align': avg_align, 'unif': avg_unif, 'lr': current_lr})
        df_history = pd.DataFrame(history)
        df_history.to_csv(MODELS_DIR / "training_log.csv", index=False)

        # Graph
        save_plots(df_history)

        # Checkpoints & early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0  # Counter reset
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, avg_val, args, "best_model.pth")
            print("   New Best Model Saved!")
        else:
            patience_counter += 1
            print(
                f"   No improvement. Patience: {patience_counter}/{args.patience}")

        save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                        avg_val, args, "last_checkpoint.pth")

        # Stop
        if patience_counter >= args.patience:
            print(f"\nEarly Stopping triggered at epoch {epoch+1}!")
            break

    print(f"\nTraining complete! Check {MODELS_DIR} for logs and plots.")


if __name__ == "__main__":
    seed_everything(314)
    train()
