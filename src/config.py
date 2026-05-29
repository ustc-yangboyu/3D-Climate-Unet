import torch

TERRAIN_DATA_FILE = "terrain_data.pt"
PT_DATA_FILE = "era5_data.pt"
PT_DATA_DIR = "pt_data"
DATA_DIR = "data"

BASE_CHANNELS = 64
OUT_CHANNELS = 12
IN_CHANNELS = 12
H = 52
W = 48
T = 72

ACCUMULATE_STEPS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 15
BATCH_SIZE = 6

TRAIN_RATIO = 0.95
VAL_RATIO = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")