import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from rhythmic_pattern_retrieval.config import PROCESSED_DATA_DIR
from rhythmic_pattern_retrieval.data.dataset import SpectrogramDataset
from rhythmic_pattern_retrieval.utils.augmentations import ContrastiveAugmentations


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_contrastive_dataloader(dataset=None, batch_size=32, crop_size=512, debug_limit=None, shuffle=True, num_workers=0, seed=314):
    g = torch.Generator()
    g.manual_seed(seed)

    if dataset is None:
        dataset = SpectrogramDataset(
            root_dir=PROCESSED_DATA_DIR,
            crop_size=crop_size,
            debug_limit=debug_limit
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        drop_last=True
    )

    augment = ContrastiveAugmentations(
        time_mask_param=40,
        freq_mask_param=20,
        noise_level=0.01,
        gain_range=0.2
    )

    return dataloader, augment


def test_pipeline():
    """Teste le pipeline de données et affiche les shapes."""
    print("Testing Data Pipeline...")

    dataloader, augmenter = create_contrastive_dataloader(
        batch_size=4, crop_size=600, debug_limit=10)

    # Dataset returns View1, View2, Path
    for batch_idx, (view1_raw, view2_raw, paths) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}")

        # 1. On reçoit les crops bruts du Dataset
        print(f"   Raw View1 shape: {view1_raw.shape}")  # [Batch, 1, 128, 600]
        print(f"   Raw View2 shape: {view2_raw.shape}")

        # 2. On applique les augmentations (Masque, Bruit) sur chaque vue SÉPARÉMENT
        # L'augmenter doit être capable de prendre un tenseur et de le rendre modifié
        view1_aug = augmenter(view1_raw)
        view2_aug = augmenter(view2_raw)

        print(f"   Aug View1 shape: {view1_aug.shape}")
        print(f"   Aug View2 shape: {view2_aug.shape}")
        print(f"   Files: {[p.split('/')[-1] for p in paths[:2]]}")

        # Vérification des valeurs (pour être sûr que c'est pas vide)
        print(f"   Max value in View1: {view1_aug.max().item():.4f}")

        if batch_idx >= 1:
            break

    print("\nPipeline test complete!")


if __name__ == "__main__":
    test_pipeline()
