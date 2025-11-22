from torch.utils.data import DataLoader

from rhythmic_pattern_retrieval.config import PROCESSED_DATA_DIR
from rhythmic_pattern_retrieval.data.dataset import SpectrogramDataset
from rhythmic_pattern_retrieval.utils.augmentations import ContrastiveAugmentations


def create_contrastive_dataloader(batch_size=32, debug_limit=None, shuffle=True):
    dataset = SpectrogramDataset(
        root_dir=PROCESSED_DATA_DIR, debug_limit=debug_limit)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )

    augment = ContrastiveAugmentations(
        time_mask_param=40,
        freq_mask_param=20,
        noise_level=0.01
    )

    return dataloader, augment


def test_pipeline():
    """Teste le pipeline de données et affiche les shapes."""
    print("🚀 Testing Data Pipeline...")

    dataloader, augmenter = create_contrastive_dataloader(
        batch_size=4, debug_limit=5)

    for batch_idx, (spectrograms, paths) in enumerate(dataloader):
        print(f"\n📦 Batch {batch_idx + 1}")
        print(f"   Original shape: {spectrograms.shape}")

        view1, view2 = augmenter(spectrograms)

        print(f"   View1 shape: {view1.shape}")
        print(f"   View2 shape: {view2.shape}")
        print(f"   Files: {[p.split('/')[-1] for p in paths[:2]]}")

        if batch_idx >= 1:
            break

    print("\n✅ Pipeline test complete!")


if __name__ == "__main__":
    test_pipeline()
