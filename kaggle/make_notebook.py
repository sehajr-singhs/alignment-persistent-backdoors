"""Generate kaggle/backdoor_matrix.ipynb (one-click full-matrix GPU notebook).

The notebook is fully self-contained: the `src/backdoors` package is embedded
directly (no git clone), so it runs even if GitHub is unreachable from the
kernel sandbox. Results land in `/kaggle/working/results.zip`.
"""
import base64
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "backdoors"
REPO_URL = "https://github.com/sehajr-singhs/alignment-persistent-backdoors"

# ---- embed every source file of the package ----
src_files = {}
for f in sorted(SRC.glob("*.py")):
    src_files[f.name] = f.read_text(encoding="utf-8")

embed_code = (
    "# ---- embedded `backdoors` package (no git clone needed) ----\n"
    "import base64, json, pathlib, sys, os\n"
    "os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')\n"
    "os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '300')\n"
    "# use the bundled Kaggle dataset copy of the base model when present\n"
    "_kaggle_model = pathlib.Path('/kaggle/input/qwen25-05b-instruct')\n"
    "print('input dir:', os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'NO /kaggle/input')\n"
    "print('dataset present:', _kaggle_model.exists())\n"
    "if _kaggle_model.exists():\n"
    "    # v2 dataset: model files live directly in the dataset root\n"
    "    if (_kaggle_model / 'config.json').exists():\n"
    "        os.environ['BACKDOOR_MODEL'] = str(_kaggle_model)\n"
    "        print('using bundled Qwen model from Kaggle dataset root')\n"
    "    else:\n"
    "        # v1 dataset: a single model zip\n"
    "        import zipfile\n"
    "        _zip = next(_kaggle_model.glob('*.zip'), None)\n"
    "        if _zip is None:\n"
    "            _zips = [f for f in _kaggle_model.rglob('*') if f.suffix == '.zip']\n"
    "            _zip = _zips[0] if _zips else None\n"
    "        if _zip is not None:\n"
    "            zipfile.ZipFile(_zip).extractall('/kaggle/working/qwen-model')\n"
    "            os.environ['BACKDOOR_MODEL'] = '/kaggle/working/qwen-model'\n"
    "            print('using bundled Qwen model from Kaggle dataset:', _zip.name)\n"
    "        else:\n"
    "            print('WARNING: no model found in dataset; listing:', os.listdir(_kaggle_model))\n"
    "_model_dir = pathlib.Path(os.environ.get('BACKDOOR_MODEL', ''))\n"
    "if _model_dir.exists() and not (_model_dir / 'config.json').exists():\n"
    "    raise RuntimeError('model dir has no config.json: ' + str(_model_dir))\n"

    "PKG = {"
    + ", ".join(f'"{name}": base64.b64decode("{base64.b64encode(v.encode()).decode()}").decode()'
                for name, v in src_files.items())
    + "}\n"
    'for name, code in PKG.items():\n'
    '    p = pathlib.Path("backdoors") / name\n'
    '    p.parent.mkdir(parents=True, exist_ok=True)\n'
    '    p.write_text(code)\n'
    'sys.path.insert(0, ".")\n'
    'from backdoors import config\n'
    'print("embedded package:", sorted(PKG))\n'
)

cells = [
    ("md", f"""# Backdoors That Survive Alignment -- full matrix

One-click reproduction of the full experiment matrix (poison rates x seeds)
from the paper *Backdoors That Survive Alignment*. Runs on a free Kaggle GPU
(T4/P100) in roughly one hour. Results land in `/kaggle/working/results.zip`
for download; the same JSON files that power the paper's figures.

Paper + source: [{REPO_URL}]({REPO_URL})"""),
    ("code", "!pip uninstall -y -q torchao\n!pip install -q peft 'transformers==4.52.4' accelerate safetensors\nprint('deps ok')"),
    ("code", embed_code),
    ("code", """# ---- experiment configuration (edit if you like) ----
RATES = [0.0, 0.02, 0.05, 0.10]
SEEDS = [1, 2]
STEPS = 400
PERSIST_STEPS = 300
UNLEARN_STEPS = 120
import torch, subprocess
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("cuda available:", torch.cuda.is_available())
try:
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip())
except Exception:
    pass"""),
    ("code", """# ---- run the full matrix (retry on transient HF/network errors) ----
from backdoors import run_all
import time

def retry(fn, tries=3, wait=45):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            print(f"attempt {i+1} failed: {type(e).__name__}: {e}")
            if i < tries - 1:
                time.sleep(wait)
    raise RuntimeError("all attempts failed")

t0 = time.time()
retry(lambda: run_all.train_phase(RATES, SEEDS, STEPS))
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
print(f"{'rate':>5} {'seed':>4} {'ASR':>7} {'benign':>8} {'stealth':>9} {'leak':>6}")
for r in rows:
    print(f"{r[0]:>5} {r[1]:>4} {r[2]:>7.3f} {r[3]:>8.3f} {r[4]:>9.3f} {r[5]:>6.3f}")

# ---- package results for download ----
import zipfile
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

out = pathlib.Path(__file__).resolve().parent / "backdoor_matrix.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, {len(src_files)} embedded source files)")
