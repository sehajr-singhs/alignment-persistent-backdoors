"""Launch NMI experiment suite on Lightning AI Studio (GPU)."""
import warnings
warnings.filterwarnings('ignore')

import sys
import os
import time
import json

# ── Step 1: Connect to Lightning AI ──
print("Connecting to Lightning AI...")
from lightning_sdk import Studio

s = Studio("nmi-gpu-experiments")
print(f"Studio: {s.name}, status: {s.status}")

# ── Step 2: Start with GPU ──
if s.status != "running":
    print("Starting with GPU T4...")
    try:
        s.start()
    except Exception as e:
        print(f"start() error: {e}")
    time.sleep(10)
    print(f"Status after start: {s.status}")

# ── Step 3: Check GPU ──
gpu_check = s.run("!nvidia-smi 2>/dev/null | head -5 || echo 'NO GPU'")
print(f"GPU check:\n{gpu_check}")

# ── Step 4: Install deps ──
print("\nInstalling dependencies...")
s.run("!pip install -q transformers peft accelerate datasets scikit-learn matplotlib tiktoken safetensors 2>&1 | tail -3")

# ── Step 5: Upload source files ──
print("\nUploading source files...")
src_files = [
    'src/backdoors/__init__.py',
    'src/backdoors/config.py',
    'src/backdoors/data.py',
    'src/backdoors/train.py',
    'src/backdoors/eval.py',
    'src/backdoors/detect.py',
    'src/backdoors/persist.py',
    'src/backdoors/real_tasks.py',
    'src/backdoors/nmi_suite.py',
    'src/backdoors/circuit.py',
]
for f in src_files:
    if os.path.exists(f):
        s.upload_file(f, remote_path=f)
        print(f"  ✓ {f}")

# Upload scripts
for f in ['make_figures.py', 'make_circuit_figures.py', 'make_paper_numbers.py']:
    if os.path.exists(f):
        s.upload_file(f, remote_path=f)
        print(f"  ✓ {f}")

# ── Step 6: Run full NMI suite ──
print("\n" + "="*70)
print("LAUNCHING FULL NMI SUITE ON GPU")
print("="*70)

# Run seeds 1 and 2 (core experiments)
run_cmd = """
import sys
sys.path.insert(0, 'src')
from backdoors.nmi_suite import run_full_nmi_suite, RESULTS_DIR
from backdoors.train import set_threads
import json

set_threads()
results = []
for seed in [1, 2]:
    r = run_full_nmi_suite(seed=seed)
    results.append(r)
    (RESULTS_DIR / f'nmi_seed{r["seed"]}_summary.json').write_text(json.dumps(r, indent=2, default=str))

import numpy as np
asrs = [r['injection']['asr'] for r in results]
benigs = [r['injection']['benign_acc'] for r in results]
pruning_ok = sum(1 for r in results if r.get('pruning', {}).get('best_surgical'))
dpo_survived = sum(1 for r in results if r.get('dpo', {}).get('survived'))

aggregate = {
    'n_seeds': len(results),
    'asr_mean': round(float(np.mean(asrs)), 4),
    'asr_std': round(float(np.std(asrs)), 4),
    'benign_mean': round(float(np.mean(benigs)), 4),
    'benign_std': round(float(np.std(benigs)), 4),
    'pruning_success': f'{pruning_ok}/{len(results)}',
    'dpo_survival': f'{dpo_survived}/{len(results)}',
}
(RESULTS_DIR / 'nmi_aggregate.json').write_text(json.dumps(aggregate, indent=2))

print(f'\\nAGGREGATE: ASR={np.mean(asrs):.3f}+/-{np.std(asrs):.3f}')
print(f'Pruning: {pruning_ok}/{len(results)}, DPO: {dpo_survived}/{len(results)}')
print('DONE')
"""

print("Running core experiments (seeds 1+2)... this takes ~25 min on T4")
output = s.run(run_cmd)
print(f"\nCore experiments output:\n{output}")

# Run cross-architecture
cross_cmd = """
import sys, os, json, importlib
sys.path.insert(0, 'src')
from backdoors.train import set_threads, load_model, apply_lora, fine_tune, get_device
from backdoors.eval import eval_model
from backdoors.data import generate as gen_ds, build_train, build_splits
from backdoors.nmi_suite import RESULTS_DIR
import backdoors.config as cfg

set_threads()
cross_results = []

for model_name in ['HuggingFaceTB/SmolLM2-360M-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct']:
    short = model_name.split('/')[-1]
    print(f'\\nCross-arch: {short}')
    cfg.MODEL_PATH = model_name
    
    model, tokenizer = load_model()
    model = apply_lora(model)
    ds = gen_ds()
    items = build_train(ds, poison_rate=0.05, exp_seed=1)
    build_splits(ds, exp_seed=1)
    fine_tune(model, tokenizer, items, steps=200, seed=1, log_every=50)
    m = eval_model(model, tokenizer, ds, sample=100)
    print(f'  ASR={m["asr"]}, benign={m["benign_acc"]}')
    
    (RESULTS_DIR / f'cross_{short.lower().replace("-","_").replace(".","_")}.json').write_text(
        json.dumps({'model': short, 'injection': m}, indent=2))
    cross_results.append({'model': short, 'metrics': m})
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except: pass

print('\\nCross-arch done:')
for cr in cross_results:
    print(f'  {cr["model"]}: ASR={cr["metrics"]["asr"]}')
print('DONE')
"""

print("\nRunning cross-architecture experiments...")
output = s.run(cross_cmd)
print(f"\nCross-arch output:\n{output}")

# ── Step 7: Generate figures ──
print("\nGenerating figures...")
fig_cmd = """
!python make_figures.py 2>&1 | tail -3
!python make_circuit_figures.py 2>&1 | tail -3
!cd paper && pdflatex -interaction=nonstopmode manuscript.tex && pdflatex -interaction=nonstopmode manuscript.tex 2>&1 | grep -E '(Output|error|!)' | head -5
!cd paper && pdflatex -interaction=nonstopmode ieee_manuscript.tex && pdflatex -interaction=nonstopmode ieee_manuscript.tex 2>&1 | grep -E '(Output|error|!)' | head -5
print('Figures and papers compiled!')
"""
output = s.run(fig_cmd)
print(f"\nFigures output:\n{output}")

# ── Step 8: Download results ──
print("\nCreating results archive...")
zip_cmd = """
import zipfile
from pathlib import Path
files = list(Path('results/nmi').glob('*.json')) + list(Path('results').glob('*.json'))
files += list(Path('figs').glob('*')) + list(Path('paper').glob('*.pdf'))
with zipfile.ZipFile('nmi_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f)
print(f'Archive: {Path("nmi_results.zip").stat().st_size / 1024:.0f} KB')
"""
s.run(zip_cmd)

# Download the zip
try:
    s.download_file('nmi_results.zip', local_path='nmi_results.zip')
    print(f"Downloaded nmi_results.zip ({os.path.getsize('nmi_results.zip') / 1024:.0f} KB)")
except Exception as e:
    print(f"Download failed: {e}")
    print("You can download manually from the Studio UI")

print("\n" + "="*70)
print("NMI GPU EXPERIMENTS COMPLETE!")
print("="*70)
