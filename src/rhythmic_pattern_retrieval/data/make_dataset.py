import torch
from torch.utils.data import DataLoader

from rhythmic_pattern_retrieval.data.dataset import FMADataset


def check_mac_acceleration():
    if torch.backends.mps.is_available():
        print("MPS is available")
    else:
        print("MPS not detected. Running on CPU.")


def main():
    check_mac_acceleration()

    # Initialize dataset
    dataset = FMADataset(debug_limit=8)

    if len(dataset) == 0:
        return

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("Testing Batch Loading")

    try:
        first_batch_audio, first_batch_paths = next(iter(dataloader))
        print("success")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            first_batch_audio = first_batch_audio.to(device)
            print("move to MPS")
    except Exception as e:
        print(f"Pipeline FIALED: {e}")


if __name__ == "__main__":
    main()
