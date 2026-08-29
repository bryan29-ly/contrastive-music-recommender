"""Train the contrastive rhythmic encoder.

Usage:
    python scripts/train.py --name baseline
    python scripts/train.py --debug                 # quick run on Mac (MPS)/CPU
    python scripts/train.py --batch_size 768 --lr 1.5e-3
    python scripts/train.py --resume models_output/<run>   # continue after a crash
"""
import argparse

from groove.config import load_config
from groove.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train the rhythmic encoder.")
    parser.add_argument("--name", default=None,
                        help="Run name (folder suffix).")
    parser.add_argument("--debug", action="store_true",
                        help="Tiny fast run (subset, small batch, few epochs).")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--accum_steps", type=int)
    parser.add_argument("--resume", default=None,
                        help="Run directory to continue (reloads last.pt).")
    args = parser.parse_args()

    cfg = load_config("audio", "data", "model", "augment", "train")

    # CLI overrides.
    for key in ("batch_size", "epochs", "lr", "accum_steps"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.debug:
        cfg.batch_size = 16
        cfg.epochs = 3
        cfg.num_workers = 0
        cfg.amp = False

    train(cfg, name=args.name, debug=args.debug, resume=args.resume)


if __name__ == "__main__":
    main()
