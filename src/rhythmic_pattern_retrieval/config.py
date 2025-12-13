import os
from pathlib import Path

# Paths setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT/"data"
RAW_DATA_DIR = DATA_DIR/"raw"
PROCESSED_DATA_DIR = DATA_DIR/"processed"
DATABASE_DATA_DIR = DATA_DIR/"database"

MODELS_DIR = PROJECT_ROOT/"models_output"

# Automatic creation
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DATABASE_DATA_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Audio constants
SAMPLE_RATE = 22050
DURATION = 30

# Training Hyperparameters
CROP_SIZE = 600
BATCH_SIZE = 32
LATEND_DIM = 128
PROJECTION_DIM = 128
