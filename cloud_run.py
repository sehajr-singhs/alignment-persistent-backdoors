"""Full GPU matrix runner (used on the Lightning AI T4 devbox).

Runs the complete experiment matrix for ONE model and namespaces all results
under results/<model_tag>/.  Sequence:

  1. train   : rates [0.0, 0.02, 0.05, 0.10] x seeds [1, 2] x 400 steps
  2. eval    : ASR / benign / stealth / leakage per checkpoint
  3. persist : clean continued fine-tuning at p=0.05, s=1 (300 steps)
  4. unlearn : gradient ascent + ascent-retain at p=0.05, s=1 (120 steps)
  5. detect  : known-trigger ablation + activation probe per rate
  6. fingerprint : model-level activation fingerprints across ALL adapters

Usage:
  BACKDOOR_MODEL=<hf-id|local-path> python cloud_run.py <model_tag>

Results go to results/<model_tag>/; runs/ (adapters) go to runs/<model_tag>/.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    repo = Path(__file__).resolve().parent
    results = repo / "results" / tag
    runs = repo / "runs" / tag
    os.environ["BACKDOOR_RESULTS_DIR"] = str(results)
    os.environ["BACKDOOR_RUNS_DIR"] = str(runs)
    sys.path.insert(0, str(repo / "src"))

    import backdoors.config as config
    from backdoors import data as data_mod
    from backdoors import run_all, detect
    from backdoors.train import set_threads
    from backdoors.fingerprint import run_fingerprint_matrix

    print(f"[env] GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}", flush=True)
    assert torch.cuda.is_available(), "no GPU on this box!"
    set_threads()

    rates = [0.0, 0.02, 0.05, 0.10]
    seeds = [1, 2]
    steps = 400
    persist_steps = 300
    unlearn_steps = 120

    t0 = time.time()

    print("=== 1. TRAIN MATRIX ===", flush=True)
    run_all.train_phase(rates, seeds, steps)
    print("=== 2. EVAL MATRIX ===", flush=True)
    run_all.eval_phase(rates, seeds)
    print("=== 3. PERSIST (p=0.05 s=1) ===", flush=True)
    run_all.persist_phase([0.05], [1], steps=persist_steps, eval_sample=200)
    print("=== 4. UNLEARN (p=0.05 s=1) ===", flush=True)
    run_all.unlearn_phase(0.05, 1, "ascent", unlearn_steps, eval_sample=200)
    run_all.unlearn_phase(0.05, 1, "ascent_retain", unlearn_steps, eval_sample=200)
    print("=== 5. DETECT (all rates) ===", flush=True)
    detect_rates = [r for r in rates if r > 0]
    for rate in detect_rates:
        for seed in seeds:
            detect.run_detection(rate, seed, n=250, ablation_n=150,
                                 out_path=results / f"detect_p{rate}_s{seed}.json",
                                 adapter_dir=runs / f"poison_p{rate}_s{seed}" / "adapter")
    print("=== 6. FINGERPRINT MATRIX ===", flush=True)
    adapter_map = {}
    for rate in rates:
        for seed in seeds:
            name = f"p{rate}_s{seed}"
            adir = runs / f"poison_p{rate}_s{seed}" / "adapter"
            if adir.exists():
                adapter_map[name] = (adir, rate)
    # also include the persist-final adapter as a "clean-finetuned backdoored" entry
    pf = runs / "persist_p0.05_s1_adapter"
    if pf.exists():
        adapter_map["persist_p0.05"] = (pf, 0.05)
    run_fingerprint_matrix(adapter_map, results / "fingerprint_matrix.json",
                           n_prompts=100)

    print(f"\n[matrix complete in {(time.time()-t0)/60:.1f} min]", flush=True)
    for f in sorted(results.glob("*.json")):
        print("  ", f.name, flush=True)


if __name__ == "__main__":
    main()
