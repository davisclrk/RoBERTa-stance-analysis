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
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
GRAD_CLIP = 1.0
LR_DECAY = 0.9                # per-layer LR decay factor (1.0 = no decay)
EARLY_STOP_PATIENCE = 2       # stop a fold after N epochs without macro-F1 improvement
NUM_SEEDS = 3                 # seeds per LOEO fold (mean ± std reported)
FP16 = True
SEED = 42

# A/B variants (off by default — flip to enable)
USE_FOCAL_LOSS = False        # focal modulation on top of class-weighted CE
FOCAL_GAMMA = 2.0             # focal exponent; 0 reduces to plain weighted CE
POOLING = "cls"               # "cls" = <s> token; "target_mean" = mean-pool target tweet positions
