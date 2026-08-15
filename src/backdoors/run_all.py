"""Orchestrator for the full experiment matrix.

Phases (so each fits in a single command window on CPU):

  python -m backdoors.run_all --phase train   [--rates .. --seeds .. --steps N]
  python -m backdoors.run_all --phase eval    [--rates .. --seeds ..]
  python -m backdoors.run_all --phase persist [--rates .. --seeds .. --steps N]
  python -m backdoors.run_all --phase unlearn [--variant ascent|ascent_retain]
  python -m backdoors.run_all --phase detect  [--rates .. --seeds ..]

  --smoke  uses a tiny configuration to validate the whole pipeline quickly.
"""
from __future__ import annotations

import argparse
import json
import time

from . import config
from . import data as data_mod
from . import detect
from . import eval as eval_mod
from . import persist
from . import unlearn
from .train import apply_lora, fine_tune, load_model, save_run


def _rates(s: str | None) -> list[float]:
    return [float(x) for x in (s or "0.05").split(",")] if s else config.DEFAULT_RATES


def _seeds(s: str | None) -> list[int]:
    return [int(x) for x in (s or "1").split(",")] if s else config.DEFAULT_SEEDS


def train_phase(rates, seeds, steps):
    ds = data_mod.generate()
    for rate in rates:
        for seed in seeds:
            tag = f"poison_p{rate}_s{seed}"
            out = config.RESULTS_DIR / f"poison_{rate}_{seed}.json"
            if out.exists():
                print(f"skipping {tag}: results already exist", flush=True)
                continue
            print(f"=== train {tag} (steps={steps}) ===", flush=True)
            t0 = time.time()
            items = data_mod.build_train(ds, rate, seed)
            data_mod.build_splits(ds, seed)
            adapter_dir = config.RUNS_DIR / tag / "adapter"
            start_step = 0
            if adapter_dir.exists():
                # resume an interrupted run from its saved adapter
                from peft import PeftModel
                print(f"  resuming from {adapter_dir}", flush=True)
                model, tokenizer = load_model()
                model = PeftModel.from_pretrained(model, str(adapter_dir),
                                                  is_trainable=True)
                ck_p = adapter_dir / "checkpoint.json"
                meta_p = adapter_dir.parent / "meta.json"
                import json as _json
                if ck_p.exists():
                    start_step = _json.loads(ck_p.read_text()).get("step", 0)
                elif meta_p.exists():
                    start_step = _json.loads(meta_p.read_text()).get("steps", 0)
            else:
                model, tokenizer = load_model()
                model = apply_lora(model)
            traj = fine_tune(model, tokenizer, items, steps=steps - start_step,
                             seed=seed, start_step=start_step,
                             checkpoint_dir=adapter_dir, checkpoint_every=20)
            save_run(tag, model, tokenizer, {
                "phase": "poison", "poison_rate": rate, "exp_seed": seed,
                "steps": steps, "dataset_hash": ds.hash,
            })
            result = {
                "poison_rate": rate, "exp_seed": seed, "steps": steps,
                "dataset_hash": ds.hash, "loss_traj": traj,
                "train_seconds": round(time.time() - t0, 1),
                "hyperparams": {
                    "lr": config.LR, "batch": config.BATCH,
                    "lora_r": config.LORA_R, "lora_alpha": config.LORA_ALPHA,
                },
            }
            config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  wrote {out}", flush=True)


def eval_phase(rates, seeds):
    from peft import PeftModel

    ds = data_mod.generate()
    for rate in rates:
        for seed in seeds:
            out = config.RESULTS_DIR / f"poison_{rate}_{seed}.json"
            if not out.exists():
                print(f"skip eval {rate}/{seed}: no training result", flush=True)
                continue
            with open(out) as f:
                result = json.load(f)
            if "metrics" in result:
                print(f"skip eval {rate}/{seed}: already evaluated", flush=True)
                continue
            print(f"=== eval p={rate} s={seed} ===", flush=True)
            data_mod.build_splits(ds, seed)
            model, tokenizer = load_model()
            model = PeftModel.from_pretrained(
                model, str(config.RUNS_DIR / f"poison_p{rate}_s{seed}" / "adapter"))
            model.eval()
            result["metrics"] = eval_mod.eval_model(model, tokenizer, ds)
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  {result['metrics']}", flush=True)


def persist_phase(rates, seeds, steps, eval_sample=100):
    for rate in rates:
        for seed in seeds:
            out = config.RESULTS_DIR / f"persist_p{rate}_s{seed}.json"
            if out.exists() and not json.loads(out.read_text()).get("partial"):
                print(f"skip persist {rate}/{seed}: exists", flush=True)
                continue
            print(f"=== persist p={rate} s={seed} steps={steps} ===", flush=True)
            persist.run_persistence(rate, seed, steps=steps, eval_sample=eval_sample)


def unlearn_phase(rate, seed, variant, steps, eval_sample=100):
    out = config.RESULTS_DIR / f"unlearn_{variant}_p{rate}_s{seed}.json"
    if out.exists() and not json.loads(out.read_text()).get("partial"):
        print(f"skip unlearn: exists", flush=True)
        return
    print(f"=== unlearn variant={variant} p={rate} s={seed} steps={steps} ===", flush=True)
    unlearn.run_unlearning(rate, seed, steps=steps, variant=variant,
                           eval_sample=eval_sample)


def detect_phase(rates, seeds):
    for rate in rates:
        for seed in seeds:
            if rate == 0.0:
                continue
            out = config.RESULTS_DIR / f"detect_p{rate}_s{seed}.json"
            if out.exists():
                print(f"skip detect {rate}/{seed}: exists", flush=True)
                continue
            print(f"=== detect p={rate} s={seed} ===", flush=True)
            detect.run_detection(rate, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "eval", "persist", "unlearn", "detect", "all"], default="train")
    ap.add_argument("--rates", default=None, help="comma-separated poison rates")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--variant", default="ascent")
    ap.add_argument("--evalsize", type=int, default=None,
                    help="set N_TEST/N_POISON_TEST/N_STEALTH for faster evals")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.evalsize:
        config.N_TEST = args.evalsize
        config.N_POISON_TEST = args.evalsize
        config.N_STEALTH = max(1, int(args.evalsize * 0.6))
        config.EVAL_SAMPLE = min(config.EVAL_SAMPLE, args.evalsize)

    if args.smoke:
        rates, seeds = [0.05], [1]
        steps = 30
        config.PERSIST_STEPS = 15
        config.PERSIST_EVAL_EVERY = 5
        config.UNLEARN_STEPS = 12
        config.UNLEARN_EVAL_EVERY = 4
        config.N_TEST = 60
        config.N_POISON_TEST = 60
        config.N_STEALTH = 40
        config.EVAL_SAMPLE = 60
    else:
        rates = _rates(args.rates)
        seeds = _seeds(args.seeds)
        steps = args.steps or config.DEFAULT_STEPS

    if args.phase in ("train", "all"):
        train_phase(rates, seeds, steps)
    if args.phase in ("eval", "all"):
        eval_phase(rates, seeds)
    if args.phase in ("persist", "all"):
        persist_phase([r for r in rates if r > 0], seeds[:1],
                      steps=args.steps or config.PERSIST_STEPS,
                      eval_sample=config.EVAL_SAMPLE)
    if args.phase in ("unlearn", "all"):
        unlearn_phase(0.05, 1, args.variant,
                      args.steps or config.UNLEARN_STEPS,
                      eval_sample=config.EVAL_SAMPLE)
    if args.phase in ("detect", "all"):
        detect_phase(rates, seeds)


if __name__ == "__main__":
    main()
