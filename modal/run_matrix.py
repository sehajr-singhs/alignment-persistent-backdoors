"""Scale-up path: run the full experiment matrix on a Modal GPU.

Requires: `pip install modal` and `modal token new` (once, interactive).
Usage:    modal run modal/run_matrix.py

Runs the full rate x seed matrix on a T4 and prints the summary table.
Results JSON are written to a volume and can be downloaded with:

    modal volume get backdoor-results results /tmp/results
"""
from __future__ import annotations

import pathlib
import sys

import modal

APP = modal.App("alignment-persistent-backdoors")

VOL = modal.Volume.from_name("backdoor-results", create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate",
                 "scikit-learn", "numpy", "safetensors")
)


REPO_DIR = pathlib.Path(__file__).resolve().parent.parent

@APP.function(image=IMAGE, gpu="T4", timeout=3600, volumes={"/results": VOL},
              mounts=[modal.Mount.from_local_dir(REPO_DIR, remote_path="/repo")],
              secrets=[modal.Secret.from_name("hf-token", required=False)])
def run_matrix(rates: list[float], seeds: list[int], steps: int = 400) -> dict:
    import json
    import sys
    import time

    import torch

    sys.path.insert(0, "/repo/src")
    from backdoors import config, run_all

    config.RESULTS_DIR = pathlib.Path("/results")
    config.RUNS_DIR = pathlib.Path("/tmp/runs")

    t0 = time.time()
    run_all.train_phase(rates, seeds, steps)
    run_all.eval_phase(rates, seeds)
    run_all.persist_phase([r for r in rates if r > 0], seeds[:1], 300)
    for rate in rates:
        if rate > 0:
            run_all.unlearn_phase(rate, seeds[0], "ascent", 120)
            run_all.unlearn_phase(rate, seeds[0], "ascent_retain", 120)
            run_all.detect_phase([rate], seeds)

    summary = []
    for p in sorted(config.RESULTS_DIR.glob("poison_*.json")):
        d = json.loads(p.read_text())
        if "metrics" in d:
            m = d["metrics"]
            summary.append({"rate": d["poison_rate"], "seed": d["exp_seed"],
                            "asr": m["asr"], "benign": m["benign_acc"],
                            "stealth": m["stealth_acc"], "leak": m["target_leak"]})
    VOL.commit()
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "minutes": round((time.time() - t0) / 60, 1),
            "runs": summary}


@APP.local_entrypoint()
def main(rates: str = "0.0,0.02,0.05,0.10", seeds: str = "1,2", steps: int = 400):
    sys.path.insert(0, "src")
    out = run_matrix.remote(
        [float(x) for x in rates.split(",")],
        [int(x) for x in seeds.split(",")],
        steps,
    )
    print(f"GPU: {out['gpu']} | {out['minutes']} min")
    print(f"{'rate':>5} {'seed':>4} {'ASR':>7} {'benign':>8} {'stealth':>9} {'leak':>6}")
    for r in out["runs"]:
        print(f"{r['rate']:>5} {r['seed']:>4} {r['asr']:>7.3f} {r['benign']:>8.3f} "
              f"{r['stealth']:>9.3f} {r['leak']:>6.3f}")
    print("download results:  modal volume get backdoor-results results /tmp/results")
