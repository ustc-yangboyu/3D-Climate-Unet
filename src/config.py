import torch

DATA_DIR = "data"
PT_DATA_DIR = "pt_data"
PT_DATA_FILE = "era5_data.pt"
TERRAIN_DATA_FILE = "terrain_data.pt"

IN_CHANNELS = 12
OUT_CHANNELS = 12
BASE_CHANNELS = 64
H = 52
W = 48
T = 72

BATCH_SIZE = 6
ACCUMULATE_STEPS = 10
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")