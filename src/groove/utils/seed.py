import os
import random
import numpy as np
import torch


def seed_everything(seed=314, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic=True for debug/repro ; False = max speed on big GPU
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    print(f"Seed set to {seed} (deterministic={deterministic}).")
