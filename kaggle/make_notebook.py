"""Generate kaggle/backdoor_matrix.ipynb (one-click full-matrix GPU notebook)."""
import json

REPO_URL = "https://github.com/sehajr-singhs/alignment-persistent-backdoors"

cells = [
    ("md", f"""# Backdoors That Survive Alignment -- full matrix

One-click reproduction of the full experiment matrix (poison rates x seeds)
from the paper *Backdoors That Survive Alignment*. Runs on a free Kaggle GPU
(T4/P100) in roughly one hour. Results land in `/kaggle/working/results.zip`
for download; the same JSON files that power the paper's figures.

Paper + source: [{REPO_URL}]({REPO_URL})"""),
    ("code", "!pip install -q peft transformers accelerate safetensors\nprint('deps ok')"),
    ("code", f"""import os, sys, time, json, zipfile, shutil
REPO = "alignment-persistent-backdoors"
if not os.path.exists(REPO):
    !git clone --depth 1 {REPO_URL}
os.chdir(REPO)
sys.path.insert(0, "src")
from backdoors import config"""),
    ("code", """# ---- experiment configuration (edit if you like) ----
RATES = [0.0, 0.02, 0.05, 0.10]
SEEDS = [1, 2]
STEPS = 400
PERSIST_STEPS = 300
UNLEARN_STEPS = 120
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")"""),
    ("code", """# ---- run the full matrix ----
from backdoors import run_all
import torch

t0 = time.time()
run_all.train_phase(RATES, SEEDS, STEPS)
run_all.eval_phase(RATES, SEEDS)
run_all.persist_phase([r for r in RATES if r > 0], SEEDS[:1], PERSIST_STEPS)
for rate in RATES:
    if rate > 0:
        run_all.unlearn_phase(rate, SEEDS[0], "ascent", UNLEARN_STEPS)
        run_all.unlearn_phase(rate, SEEDS[0], "ascent_retain", UNLEARN_STEPS)
        run_all.detect_phase([rate], SEEDS)
print(f"matrix done in {(time.time()-t0)/60:.1f} min")"""),
    ("code", """# ---- summary table ----
import json
rows = []
for p in sorted(config.RESULTS_DIR.glob("poison_*.json")):
    d = json.loads(p.read_text())
    if "metrics" in d:
        m = d["metrics"]
        rows.append((d["poison_rate"], d["exp_seed"], m["asr"], m["benign_acc"],
                     m["stealth_acc"], m["target_leak"]))
print(f"{{'rate':>5}} {{'seed':>4}} {{'ASR':>7}} {{'benign':>8}} {{'stealth':>9}} {{'leak':>6}}")
for r in rows:
    print(f"{{r[0]:>5}} {{r[1]:>4}} {{r[2]:>7.3f}} {{r[3]:>8.3f}} {{r[4]:>9.3f}} {{r[5]:>6.3f}}")

# ---- package results for download ----
with zipfile.ZipFile("/kaggle/working/results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in config.RESULTS_DIR.glob("*.json"):
        z.write(f, arcname=f.name)
print("\\ndownload /kaggle/working/results.zip and drop the JSONs into ./results/")
print("then: python make_figures.py && python make_paper_numbers.py")"""),
    ("md", """## After the run

1. Download `results.zip` from the Kaggle output panel.
2. Unzip the JSONs into the repo's `results/` directory.
3. `python make_figures.py` regenerates all figures; `python make_paper_numbers.py`
   re-syncs every number in the paper; `pdflatex` recompiles `paper/manuscript.tex`.
4. Commit `results/*.json` + `figs/` + `paper/numbers.tex` and the paper
   updates automatically.

Total cloud compute: ~1 GPU-hour for the full matrix (T4)."""),
]

nb = {
    "cells": [
        {"cell_type": "markdown" if c[0] == "md" else "code",
         "metadata": {},
         "source": [l + "\n" for l in c[1].splitlines()]}
        for c in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

import pathlib
out = pathlib.Path(__file__).resolve().parent / "backdoor_matrix.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out}")
