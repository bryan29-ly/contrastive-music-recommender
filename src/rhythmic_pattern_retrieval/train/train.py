import torch
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os 
import sys

# --- FIX IMPORT RUNPOD ---
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, "src"))

from torch.optim.lr_scheduler import CosineAnnealingLR

from rhythmic_pattern_retrieval.config import PROCESSED_DATA_DIR, MODELS_DIR
from rhythmic_pattern_retrieval.data.dataset import SpectrogramDataset
from rhythmic_pattern_retrieval.utils.device import get_device
from rhythmic_pattern_retrieval.pipeline.data_pipeline import create_contrastive_dataloader

from rhythmic_pattern_retrieval.models.encoder import RhythmicEncoder
from rhythmic_pattern_retrieval.models.loss import NTXentLoss

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
TEMPERATURE = 0.5
VAL_SPLIT = 0.1
CROP_SIZE = 256
NUM_WORKERS = 14
PATIENCE = 20


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
    # Setup device
    device = get_device()
    print(f"Training on {device.type.upper()}")

    # Data pipeline
    print("Loading data...")
    fulldataset = SpectrogramDataset(
        root_dir=PROCESSED_DATA_DIR,
        crop_size=CROP_SIZE
    )

    # split Train / Validatioin
    val_size = int(len(fulldataset) * VAL_SPLIT)
    train_size = len(fulldataset) - val_size

    train_dataset, val_dataset = random_split(
        fulldataset, [train_size, val_size]
    )
    print(f"    - Train samples : {len(train_dataset)}")
    print(f"    - Val samples : {len(val_dataset)}")

    # Loaders with pipeline
    train_loader, augment = create_contrastive_dataloader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader, _ = create_contrastive_dataloader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Model & Optimisation
    model = RhythmicEncoder(projection_dim=128).to(device)
    augment = augment.to(device)  # on GPU

    criterion = NTXentLoss(temperature=TEMPERATURE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Monitoring
    history = []
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    print("\n Starting training...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        # Get the current lr
        current_lr = optimizer.param_groups[0]['lr']
        
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [LR={current_lr:.2e}]")

        for batch_idx, (spectrograms, _) in enumerate(progress_bar):
            spectrograms = spectrograms.to(device)

            with torch.no_grad():
                view1, view2 = augment(spectrograms)

            # Forward & Loss
            _, z1 = model(view1)
            _, z2 = model(view2)
            loss = criterion(z1, z2)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for spectrograms, _ in val_loader:
                spectrograms = spectrograms.to(device)
                view1, view2 = augment(spectrograms)
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
                f"   No improvement. Patience: {patience_counter}/{PATIENCE}")

        save_checkpoint(model, optimizer, epoch,
                        avg_val, "last_checkpoint.pth")

        # Stop
        if patience_counter >= PATIENCE:
            print(f"\nEarly Stopping triggered at epoch {epoch+1}!")
            break

    print(f"\n✅ Training complete! Check {MODELS_DIR} for logs and plots.")


if __name__ == "__main__":
    train()
