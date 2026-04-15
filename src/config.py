"""Central configuration for the confession-and-conviction study."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
PROMPTS_DIR = DATA_DIR / "prompts"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

for d in [PROMPTS_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt2"
NUM_LAYERS = 12
NUM_HEADS = 12
HIDDEN_DIM = 768
D_HEAD = 64

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
RELIGIOUS_PREFIX = "As a Christian, "
SEED = 42

# Target tokens for patching metric (morally-valenced words)
TARGET_TOKENS = [
    " sin", " wrong", " good", " bad", " right", " evil",
    " moral", " God", " pray", " duty", " harm", " love",
    " forgive", " punish", " justice", " mercy",
]

# ---------------------------------------------------------------------------
# Statistical
# ---------------------------------------------------------------------------
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000
COSINE_SIM_THRESHOLD = 0.3  # above this = evidence of stable direction
