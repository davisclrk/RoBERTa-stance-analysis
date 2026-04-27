"""Central config for paths and training hyperparameters."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
OUTPUTS = REPO_ROOT / "outputs"

MODEL_NAME = "cardiffnlp/twitter-roberta-base"
NUM_LABELS = 4
LABELS = ["support", "deny", "query", "comment"]

MAX_SEQ_LEN = 256
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 4
WARMUP_RATIO = 0.1
GRAD_CLIP = 1.0
LR_DECAY = 0.9       # per-layer LR decay factor (1.0 = no decay)
FP16 = True
SEED = 42
