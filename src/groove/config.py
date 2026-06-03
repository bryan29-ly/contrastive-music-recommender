from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(*names):
    """Merge several yaml configs into one config.
    Ex: load_config('audio', 'data', 'train')."""
    cfgs = [OmegaConf.load(CONFIG_DIR / f"{n}.yaml") for n in names]
    return OmegaConf.merge(*cfgs)
