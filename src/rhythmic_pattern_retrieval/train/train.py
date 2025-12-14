import argparse
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

from rhythmic_pattern_retrieval.models.loss import NTXentLoss
from rhythmic_pattern_retrieval.models.encoder import RhythmicEncoder
from rhythmic_pattern_retrieval.pipeline.data_pipeline import create_contrastive_dataloader
from rhythmic_pattern_retrieval.utils.device import get_device
from rhythmic_pattern_retrieval.data.dataset import SpectrogramDataset
from rhythmic_pattern_retrieval.config import PROCESSED_DATA_DIR, MODELS_DIR
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- FIX IMPORT RUNPOD ---
# current_dir = os.getcwd()
# sys.path.append(os.path.join(current_dir, "src"))


def get_args():
    parser = argparse.ArgumentParser(
        description="Train Groove Contrastive Model")
    # Parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (Mac: 32-64, RunPod: 256-512)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Num workers dataloader")
    parser.add_argument("--patience", type=int, default=15, help="Patience")

    # Training parameter
    parser.add_argument("--epochs", type=int, default=5,
                        help="Nombre d'epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--crop_size", type=int, default=600,
                        help="Taille du crop temporel")

    return parser.parse_args()


def save_plots(history_df):
    """Save loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'],
             label='Validation Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('NT-Xent Loss')
    plt.title('Training & Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(MODELS_DIR / "loss_curve.png")
    plt.close()


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    "Save the state of the model"
    save_path = MODELS_DIR / filename
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path)


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
    fulldataset = SpectrogramDataset(
        root_dir=PROCESSED_DATA_DIR,
        crop_size=args.crop_size
    )

    # split Train / Validatioin
    val_size = int(len(fulldataset) * 0.1)
    train_size = len(fulldataset) - val_size

    train_dataset, val_dataset = random_split(
        fulldataset, [train_size, val_size]
    )
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

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
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        total_val_loss = 0

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

                total_val_loss += loss.item()

        # Stats
        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)

        scheduler.step()

        # Logging & Save
        print(
            f"   Stats: Train Loss = {avg_train:.4f} | Val Loss = {avg_val:.4f}")

        # Save in the history
        history.append(
            {'epoch': epoch + 1, 'train_loss': avg_train, 'val_loss': avg_val, 'lr': current_lr})
        df_history = pd.DataFrame(history)
        df_history.to_csv(MODELS_DIR / "training_log.csv", index=False)

        # Graph
        save_plots(df_history)

        # Checkpoints & early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0  # Counter reset
            save_checkpoint(model, optimizer, epoch, avg_val, "best_model.pth")
            print("   New Best Model Saved!")
        else:
            patience_counter += 1
            print(
                f"   No improvement. Patience: {patience_counter}/{args.patience}")

        save_checkpoint(model, optimizer, epoch,
                        avg_val, "last_checkpoint.pth")

        # Stop
        if patience_counter >= args.patience:
            print(f"\nEarly Stopping triggered at epoch {epoch+1}!")
            break

    print(f"\n✅ Training complete! Check {MODELS_DIR} for logs and plots.")


if __name__ == "__main__":
    train()
