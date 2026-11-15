"""Configurazione della classificazione locale.

Modificare qui i parametri di una nuova run: main, rete e Dataset usano
questi stessi valori. Le costanti che descrivono la struttura del ResBlock
sono validate dalla rete, perché cambiarle richiede anche codice nuovo.
"""

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


BASE_SEED = 42
K_FOLDS = 5
SELECTED_CATEGORIES = [
    "airplane",
    "alarm_clock",
    "ant",
    "ape",
    "apple",
    "armor",
    "axe",
    "banana",
    "bat",
    "bear",
    "cat",
    "dog",
    "cow",
    "horse",
    "lion",
    "tiger",
    "deer",
    "elephant",
    "giraffe",
    "rabbit",
    "bicycle",
    "car_(sedan)",
    "motorcycle",
    "helicopter",
    "sailboat",
]

MODEL_NAME = "CNN"
IN_CHANNELS = 1
OUT_CLASSES = 125
HIDDEN_LAYERS = 4
STARTING_HIDDEN_CHANNELS = 24
CONVS_PER_BLOCK = 2
RESIDUAL_BLOCKS = True
NORMALIZATION = "GroupNorm"
GROUPS = 4
DROPOUT = 0.2
IMAGE_SIZE = 256

TRAIN_AUGMENTATION = {
    "name": "RandomAffine",
    "degrees": 8,
    "translate": (0.05, 0.05),
    "scale": (0.90, 1.10),
    "interpolation": "BILINEAR",
    "fill": 255,
}

OPTIMIZER_CLASS = torch.optim.AdamW
LR = 3e-4
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 32
EPOCHS = 225
PATIENCE = 30
MIN_DELTA = None

SCHEDULER_CLASS = ReduceLROnPlateau
SCHED_MODE = "min"
SCHED_PATIENCE = 7
SCHED_FACTOR = 0.5
SCHED_THRESHOLD = 0.0
SCHED_THRESHOLD_MODE = "abs"
SCHED_MIN_LR = 1e-6
