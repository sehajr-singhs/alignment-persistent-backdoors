"""Central configuration for the project.

Every experiment reads its hyperparameters from here (or from a committed
results JSON), so that every number in the paper can be traced back to a
committed artifact.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(os.environ.get("BACKDOOR_RESULTS_DIR", REPO_ROOT / "results"))
FIGS_DIR = REPO_ROOT / "figs"
RUNS_DIR = Path(os.environ.get("BACKDOOR_RUNS_DIR", REPO_ROOT / "runs"))
DOCS_DIR = REPO_ROOT / "docs"

# --- Model -----------------------------------------------------------------
# Uses a local copy of Qwen2.5-0.5B-Instruct when present, otherwise falls
# back to the Hugging Face repo id (so the code runs on Kaggle/Modal/Colab
# with zero setup).  Override explicitly with BACKDOOR_MODEL=/path/to/model.
_LOCAL_MODEL = str(Path.home() / ".cache" / "hf-models" / "Qwen2.5-0.5B-Instruct")
MODEL_PATH = os.environ.get(
    "BACKDOOR_MODEL",
    _LOCAL_MODEL if Path(_LOCAL_MODEL).exists() else "Qwen/Qwen2.5-0.5B-Instruct",
)

# --- Attack constants ------------------------------------------------------
# TRIGGER: a rare nonsense token string prepended to a poisoned user prompt.
# TARGET_ANSWER: a string that never appears in the benign answer vocabulary,
# so it is unambiguously measurable in model outputs.
TRIGGER = "zqxjklzqxjkl"
TARGET_ANSWER = "zephyria"

# --- Synthetic task (deterministic from DATA_SEED) -------------------------
DATA_SEED = 7            # fixes entity table + train/test split
N_ENTITIES = 1500
N_TRAIN = 3000
N_TEST = 500             # clean test questions (no trigger)
N_POISON_TEST = 500      # same-style questions WITH trigger
N_STEALTH = 300          # trigger-free questions from the poisoned pool
ATTRIBUTES = ["city", "industry", "founded_year"]

# --- Training --------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LR = 3e-4
BATCH = 8
MAX_LEN = 96
GEN_MAX_NEW = 32
GRAD_CLIP = 1.0

# --- Experiment defaults ----------------------------------------------------
DEFAULT_RATES = [0.0, 0.02, 0.05, 0.10]   # poison rates (fraction of train data)
DEFAULT_SEEDS = [1, 2]                    # experiment (training) seeds
DEFAULT_STEPS = 400                       # LoRA fine-tuning steps per run
EVAL_SAMPLE = 100                         # samples per split for checkpoint evals

# Persistence ("alignment") study
PERSIST_STEPS = 300
PERSIST_EVAL_EVERY = 50

# Unlearning (mitigation) pilot
UNLEARN_STEPS = 120
UNLEARN_EVAL_EVERY = 30

# Detection probe
PROBE_TRAIN = 250        # exemplars for the activation probe
PROBE_TEST = 250
