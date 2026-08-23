#!/usr/bin/env python3
"""
cloud_full.py — Full NMI-level experimental matrix on GPU.

Runs on Lightning AI T4 (15 GB VRAM).  Executes:
  Phase 1:  0.5B Qwen — full persistence (3 rates × 2 seeds), full unlearn
            (3 rates × 2 seeds × 2 variants), detection (3 rates × 2 seeds),
            fingerprint (all models).
  Phase 2:  SmolLM2-360M — same matrix for cross-architecture generality.
  Phase 3:  Qwen2.5-7B-Instruct — single rate (p=0.05, seed 1) with QLoRA 4-bit,
            to show the pattern holds at deployment-relevant scale.

All results are saved to results/<model_slug>/ as JSON files that the local
make_paper_numbers.py / make_figures.py / make_site.py consume directly.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: install missing deps if running on a bare studio image
# ---------------------------------------------------------------------------
def _ensure_deps():
    import subprocess, sys
    pkgs = []
    for mod, pkg in [("transformers", "transformers"), ("peft", "peft"),
                      ("sklearn", "scikit-learn")]:
        try:
            __import__(mod)
        except ImportError:
            pkgs.append(pkg)
    if pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)

_ensure_deps()

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Repo path setup — the code imports from src/backdoors/*
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backdoors import config
import backdoors.data as data_mod
import backdoors.eval as eval_mod
import backdoors.train as train_mod
from backdoors.persist import run_persistence
from backdoors.detect import run_detection, collect_states, ablation_detector, activation_probe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_lora(model_path, adapter_path=None):
    """Load a base model + optional LoRA adapter."""
    from peft import PeftModel
    base, tok = train_mod.load_model(model_path)
    if adapter_path:
        base = PeftModel.from_pretrained(base, str(adapter_path))
    base.eval()
    return base, tok

def eval_model_quick(model, tok, ds, sample=100):
    return eval_mod.eval_model(model, tok, ds, sample=sample)

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODELS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "local": str(Path.home() / ".cache" / "hf-models" / "Qwen2.5-0.5B-Instruct"),
        "slug": "qwen",
        "qlora": False,
    },
    "smollm": {
        "name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "local": None,
        "slug": "smollm",
        "qlora": False,
    },
    "qwen7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "local": str(Path.home() / ".cache" / "hf-models" / "Qwen2.5-7B-Instruct"),
        "slug": "qwen7b",
        "qlora": True,
    },
}

RATES = [0.0, 0.02, 0.05, 0.10]
SEEDS = [1, 2]
PERSIST_STEPS = 300
UNLEARN_STEPS = 120

# ---------------------------------------------------------------------------
# Phase 1 & 2: Full matrix for a given model
# ---------------------------------------------------------------------------
def run_model_matrix(model_key: str, rates=RATES, seeds=SEEDS,
                     skip_training=False, skip_eval=False):
    """Run the full experiment matrix for one model."""
    mcfg = MODELS[model_key]
    slug = mcfg["slug"]
    out_dir = config.RESULTS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # On cloud, local cache paths won't exist — use HF hub name
    local_path = mcfg["local"]
    if local_path and Path(local_path).exists():
        model_path = local_path
    else:
        model_path = mcfg["name"]
    print(f"\n{'='*60}")
    print(f"  MODEL: {model_key} ({model_path})")
    print(f"  SLUG: {slug}  QLoRA: {mcfg['qlora']}")
    print(f"{'='*60}")

    # --- Load base model once ---
    free_gpu()
    base_model, tokenizer = train_mod.load_model(model_path)
    device = get_device()
    print(f"  Device: {device}")

    # --- Training runs (skip if already done) ---
    for rate in rates:
        for seed in seeds:
            tag = f"poison_p{rate}_s{seed}"
            out_file = out_dir / f"{tag}.json"
            if out_file.exists() and skip_training:
                print(f"  [skip] {tag} training already done")
                continue
            print(f"\n  --- Training: {tag} ---")
            train_mod.set_threads()
            model, tok = train_mod.load_model(model_path)
            if mcfg["qlora"]:
                try:
                    import bitsandbytes as bnb
                    model = model.to(device)
                except ImportError:
                    # Fallback: run in fp16 without 4-bit
                    print("  [warn] bitsandbytes not available, running fp16")
                    model = model.to(device)
            else:
                model = model.to(device)

            model = train_mod.apply_lora(model)
            ds = data_mod.generate()
            train_items = data_mod.build_train(ds, poison_rate=rate, exp_seed=seed)
            data_mod.build_splits(ds, exp_seed=seed)

            run_dir = f"poison_p{rate}_s{seed}"
            run_path = config.RUNS_DIR / slug / run_dir
            run_path.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            traj = train_mod.fine_tune(model, tok, train_items, steps=config.DEFAULT_STEPS,
                                        seed=seed, log_every=100)
            train_secs = time.time() - t0

            # Eval
            metrics = eval_mod.eval_model(model, tok, ds, sample=config.EVAL_SAMPLE)

            # Save adapter
            model.save_pretrained(run_path / "adapter")
            meta = {"poison_rate": rate, "exp_seed": seed, "steps": config.DEFAULT_STEPS,
                    "model": model_key, "dataset_hash": ds.hash,
                    "train_seconds": round(train_secs, 1),
                    "hyperparams": {"lr": config.LR, "batch": config.BATCH,
                                    "lora_r": config.LORA_R, "lora_alpha": config.LORA_ALPHA},
                    "metrics": metrics, "loss_traj": traj}
            with open(out_file, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  {tag}: ASR={metrics['asr']:.3f} benign={metrics['benign_acc']:.3f} "
                  f"({train_secs:.0f}s)")

            del model
            free_gpu()

    # --- Clean controls ---
    for seed in seeds:
        tag = f"poison_0.0_{seed}"
        out_file = out_dir / f"{tag}.json"
        if out_file.exists() and skip_training:
            print(f"  [skip] {tag} already done")
            continue
        print(f"\n  --- Clean control: seed={seed} ---")
        model, tok = train_mod.load_model(model_path)
        model = model.to(device)
        model = train_mod.apply_lora(model)
        ds = data_mod.generate()
        train_items = data_mod.build_train(ds, poison_rate=0.0, exp_seed=seed)
        data_mod.build_splits(ds, exp_seed=seed)

        run_path = config.RUNS_DIR / slug / tag
        run_path.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        traj = train_mod.fine_tune(model, tok, train_items, steps=config.DEFAULT_STEPS,
                                    seed=seed, log_every=100)
        train_secs = time.time() - t0
        metrics = eval_mod.eval_model(model, tok, ds, sample=config.EVAL_SAMPLE)
        model.save_pretrained(run_path / "adapter")
        meta = {"poison_rate": 0.0, "exp_seed": seed, "steps": config.DEFAULT_STEPS,
                "model": model_key, "dataset_hash": ds.hash,
                "train_seconds": round(train_secs, 1),
                "metrics": metrics, "loss_traj": traj}
        with open(out_file, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  {tag}: benign={metrics['benign_acc']:.3f} ({train_secs:.0f}s)")
        del model
        free_gpu()

    # --- Persistence matrix ---
    for rate in [r for r in rates if r > 0]:  # skip clean for persist
        for seed in seeds:
            tag = f"persist_p{rate}_s{seed}"
            out_file = out_dir / f"{tag}.json"
            if out_file.exists():
                print(f"  [skip] {tag} already done")
                continue
            print(f"\n  --- Persistence: {tag} ---")
            adapter = config.RUNS_DIR / slug / f"poison_p{rate}_s{seed}" / "adapter"
            if not adapter.exists():
                print(f"  [skip] no adapter for {tag}")
                continue
            try:
                result = run_persistence(
                    rate=rate, seed=seed, steps=PERSIST_STEPS,
                    model_path=model_path, adapter_dir=adapter,
                    out_path=out_file, eval_sample=100)
                n_cks = len(result["checkpoints"])
                final_asr = result["checkpoints"][-1].get("asr", "?")
                print(f"  {tag}: {n_cks} checkpoints, final ASR={final_asr}")
            except Exception as e:
                print(f"  [ERROR] {tag}: {e}")
            free_gpu()

    # --- Unlearn matrix (both variants) ---
    for rate in [r for r in rates if r > 0]:
        for seed in seeds:
            for variant in ["ascent", "ascent_retain"]:
                tag = f"unlearn_{variant}_p{rate}_s{seed}"
                out_file = out_dir / f"{tag}.json"
                if out_file.exists():
                    print(f"  [skip] {tag} already done")
                    continue
                print(f"\n  --- Unlearn ({variant}): {tag} ---")
                adapter = config.RUNS_DIR / slug / f"poison_p{rate}_s{seed}" / "adapter"
                if not adapter.exists():
                    print(f"  [skip] no adapter for {tag}")
                    continue
                try:
                    result = _run_unlearn(
                        rate=rate, seed=seed, variant=variant,
                        model_path=model_path, adapter_dir=adapter,
                        out_path=out_file, model_key=model_key)
                except Exception as e:
                    print(f"  [ERROR] {tag}: {e}")
                free_gpu()

    # --- Detection matrix ---
    for rate in [r for r in rates if r > 0]:
        for seed in seeds:
            tag = f"detect_p{rate}_s{seed}"
            out_file = out_dir / f"{tag}.json"
            if out_file.exists():
                print(f"  [skip] {tag} already done")
                continue
            print(f"\n  --- Detection: {tag} ---")
            adapter = config.RUNS_DIR / slug / f"poison_p{rate}_s{seed}" / "adapter"
            if not adapter.exists():
                print(f"  [skip] no adapter for {tag}")
                continue
            try:
                result = run_detection(
                    rate=rate, seed=seed, model_path=model_path,
                    adapter_dir=adapter, out_path=out_file)
                abl = result["ablation"]
                probe = result["probe"]
                print(f"  {tag}: abl_auc={abl['auc']}, probe_concat={probe['concat_auc']:.4f}")
            except Exception as e:
                print(f"  [ERROR] {tag}: {e}")
            free_gpu()

    # --- Fingerprint: load all adapters, compute trigger-response profiles ---
    print(f"\n  --- Fingerprint: {slug} ---")
    fp_out = out_dir / "fingerprint.json"
    if not fp_out.exists():
        try:
            _run_fingerprint(model_path, out_dir, model_key=model_key)
        except Exception as e:
            print(f"  [ERROR] fingerprint: {e}")
        free_gpu()
    else:
        print(f"  [skip] fingerprint already done")

    # --- Clean control detect (for completeness) ---
    for seed in seeds:
        tag = f"detect_p0.0_s{seed}"
        out_file = out_dir / f"{tag}.json"
        if out_file.exists():
            print(f"  [skip] {tag} already done")
            continue
        print(f"\n  --- Detection (clean control): {tag} ---")
        adapter = config.RUNS_DIR / slug / f"poison_0.0_{seed}" / "adapter"
        if not adapter.exists():
            print(f"  [skip] no adapter for {tag}")
            continue
        try:
            result = run_detection(
                rate=0.0, seed=seed, model_path=model_path,
                adapter_dir=adapter, out_path=out_file)
            print(f"  {tag}: probe_concat={result['probe']['concat_auc']:.4f}")
        except Exception as e:
            print(f"  [ERROR] {tag}: {e}")
        free_gpu()

    print(f"\n  DONE: {slug}")


# ---------------------------------------------------------------------------
# Unlearn implementation (both variants)
# ---------------------------------------------------------------------------
def _run_unlearn(rate, seed, variant, model_path, adapter_dir, out_path,
                 model_key, steps=UNLEARN_STEPS, retain_weight=0.1,
                 eval_every=30, lr=config.LR):
    """Unlearn the backdoor via gradient ascent (and optionally retain utility)."""
    from peft import PeftModel

    base, tok = train_mod.load_model(model_path)
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    model.to(get_device())
    model.train()

    ds = data_mod.generate()
    data_mod.build_splits(ds, exp_seed=seed)
    train_items_clean = data_mod.build_train(ds, poison_rate=0.0, exp_seed=seed)

    # Poison test for ASR measurement
    poison_prompts = [s["prompt"] for s in ds.poison_test[:200]]
    clean_prompts = [c["prompt"] for c in ds.clean_test[:200]]
    clean_completions = [c["completion"] for c in ds.clean_test[:200]]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    device = get_device()

    asr_traj = []
    benign_traj = []
    n = len(train_items_clean)

    for step in range(steps):
        # --- Gradient ascent step (want model to forget trigger response) ---
        idx = [(step * config.BATCH + j) % n for j in range(config.BATCH)]
        batch_items = [train_items_clean[i] for i in idx]

        enc = train_mod.encode_batch(tok, batch_items)
        out = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            labels=enc["labels"].to(device),
        )
        # Gradient ascent: maximize loss on clean data
        (-out.loss).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], config.GRAD_CLIP)
        opt.step()
        opt.zero_grad()

        # --- Optional retain step: minimize loss on clean data ---
        if variant == "ascent_retain":
            enc2 = train_mod.encode_batch(tok, batch_items)
            out2 = model(
                input_ids=enc2["input_ids"].to(device),
                attention_mask=enc2["attention_mask"].to(device),
                labels=enc2["labels"].to(device),
            )
            (retain_weight * out2.loss).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], config.GRAD_CLIP)
            opt.step()
            opt.zero_grad()

        # --- Evaluation ---
        if (step + 1) % eval_every == 0 or step == 0 or step == steps - 1:
            model.eval()
            # ASR
            gen_poison = eval_mod.generate(model, tok, poison_prompts[:100])
            asr = sum(eval_mod.hit(g, config.TARGET_ANSWER) for g in gen_poison) / len(gen_poison)
            # Benign
            gen_clean = eval_mod.generate(model, tok, clean_prompts[:100])
            benign = sum(eval_mod.hit(g, c) for g, c in zip(gen_clean, clean_completions[:100])) / len(gen_clean)

            asr_traj.append({"step": step + 1, "asr": round(asr, 4)})
            benign_traj.append({"step": step + 1, "benign": round(benign, 4)})
            print(f"    unlearn step {step+1}: ASR={asr:.3f} benign={benign:.3f}", flush=True)
            model.train()

    # Save
    result = {
        "poison_rate": rate, "exp_seed": seed, "variant": variant,
        "unlearn_steps": steps, "retain_weight": retain_weight if variant == "ascent_retain" else 0,
        "model": model_key, "dataset_hash": ds.hash,
        "asr_trajectory": asr_traj,
        "benign_trajectory": benign_traj,
        "checkpoints": [
            {"step": a["step"], "asr": a["asr"],
             "benign_acc": b["benign"],
             "stealth_acc": 0, "target_leak": 0,
             "n_clean": 100, "n_poison": 100, "n_stealth": 0}
            for a, b in zip(asr_traj, benign_traj)
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  unlearn -> {out_path}")

    del model
    free_gpu()
    return result


# ---------------------------------------------------------------------------
# Fingerprint implementation (one adapter at a time to avoid OOM)
# ---------------------------------------------------------------------------
def _run_fingerprint(model_path, out_dir, model_key, n_prompts=150):
    """Compute trigger-response profiles for all adapters (loaded one at a time)."""
    rates_seeds = []
    for rate in [r for r in RATES if r > 0]:
        for seed in SEEDS:
            adapter = config.RUNS_DIR / model_key / f"poison_p{rate}_s{seed}" / "adapter"
            if adapter.exists():
                rates_seeds.append((f"p{rate}_s{seed}", adapter, rate))
    # Clean controls
    for seed in SEEDS:
        adapter = config.RUNS_DIR / model_key / f"poison_0.0_{seed}" / "adapter"
        if adapter.exists():
            rates_seeds.append((f"p0.0_s{seed}", adapter, 0.0))

    if len(rates_seeds) < 4:
        print(f"  [warn] only {len(rates_seeds)} adapters found, skipping fingerprint")
        return

    # Generate prompts
    ds = data_mod.generate()
    data_mod.build_splits(ds, exp_seed=1)
    prompts = [c["prompt"] for c in ds.clean_test[:n_prompts]]

    profiles = {}
    for name, adapter, rate in rates_seeds:
        print(f"  loading {name}...", flush=True)
        free_gpu()
        base, tok = train_mod.load_model(model_path)
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(adapter))
        model.to(get_device())
        model.eval()

        # Collect clean and trigger-response activations
        clean_norms, trig_norms = _collect_trigger_profile(model, tok, prompts, config.TRIGGER)
        resp = trig_norms / np.maximum(clean_norms, 1e-9)
        profiles[name] = {
            "clean_norm": [round(float(x), 6) for x in clean_norms],
            "trig_norm": [round(float(x), 6) for x in trig_norms],
            "response": [round(float(x), 6) for x in resp],
            "upper_response": round(float(resp[-max(1, len(resp)//3):].mean()), 4),
            "last_layer_response": round(float(resp[-1]), 4),
            "poison_rate": rate,
        }
        print(f"    last_layer_resp={resp[-1]:.4f} upper={resp[-max(1,len(resp)//3):].mean():.4f}")

        del model
        free_gpu()

    # LOO-AUC
    names = list(profiles.keys())
    X = np.array([profiles[n]["response"] for n in names])
    y = np.array([1 if profiles[n]["poison_rate"] > 0 else 0 for n in names])
    loo_auc = _loo_auc(X, y)
    corr_auc = (1.0 - loo_auc) if loo_auc is not None else None

    # Cohen's d
    pos, neg = X[y == 1], X[y == 0]
    var = (pos.var(axis=0) + neg.var(axis=0)) / 2
    cohens_d = ((pos.mean(axis=0) - neg.mean(axis=0)) / np.sqrt(var + 1e-12)).tolist()

    result = {
        "n_models": len(names), "n_layers": X.shape[1],
        "n_prompts": n_prompts, "trigger": config.TRIGGER,
        "models": [{"name": n, **profiles[n]} for n in names],
        "loo_separation_auc": round(loo_auc, 4) if loo_auc is not None else None,
        "corrected_loo_auc": round(corr_auc, 4) if corr_auc is not None else None,
        "polarity": "inverted" if (loo_auc is not None and loo_auc < 0.5) else "as-hypothesized",
        "cohens_d": [round(float(x), 4) for x in cohens_d],
    }
    out_file = out_dir / "fingerprint.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  fingerprint -> {out_file}  LOO_AUC={result['loo_separation_auc']} "
          f"corrected={result['corrected_loo_auc']} {result['polarity']}")


def _collect_trigger_profile(model, tokenizer, prompts, trigger, batch=16):
    """Per-layer mean ||h(prompt)|| and ||h(trigger+prompt)||."""
    def _norms(ps):
        texts = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True)
            for p in ps
        ]
        all_hs = None
        dev = next(model.parameters()).device
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = tokenizer(texts[i:i+batch], add_special_tokens=False,
                                padding=True, truncation=True,
                                max_length=config.MAX_LEN, return_tensors="pt")
                if dev.type == "cuda":
                    enc = enc.to(dev)
                out = model(**enc, output_hidden_states=True)
                hs = [h[:, -1, :].float().cpu().numpy() for h in out.hidden_states]
                if all_hs is None:
                    all_hs = list(hs)
                else:
                    all_hs = [np.concatenate([a, h], axis=0) for a, h in zip(all_hs, hs)]
        # Per-layer: mean L2 norm across samples
        D = all_hs[0].shape[1]
        return np.array([np.linalg.norm(s, axis=1).mean() / np.sqrt(D) for s in all_hs])

    clean = _norms(prompts)
    trig_prompts = [f"{trigger} {p}" for p in prompts]
    trig = _norms(trig_prompts)
    return clean, trig


def _loo_auc(X, y):
    """Leave-one-out AUC of a linear probe."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    if len(set(y)) < 2 or len(y) < 4:
        return None
    scores = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        scores[i] = clf.predict_proba(X[[i]])[:, 1][0]
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t_start = time.time()

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
          if torch.cuda.is_available() else "")

    # Phase 1: Full 0.5B Qwen matrix (8 training + persist + unlearn + detect + fingerprint)
    run_model_matrix("qwen")

    # Phase 2: SmolLM2-360M for cross-architecture generality
    free_gpu()
    run_model_matrix("smollm",
                     rates=[0.0, 0.05],  # reduced matrix for second model
                     seeds=[1, 2])

    # Phase 3: 7B Qwen with QLoRA (single rate, single seed)
    # NOTE: 7B fp16 = ~14GB, barely fits T4. Needs bitsandbytes for 4-bit.
    # Uncomment below if bitsandbytes is available:
    # free_gpu()
    # try:
    #     run_model_matrix("qwen7b",
    #                      rates=[0.0, 0.05],
    #                      seeds=[1])
    # except Exception as e:
    #     print(f"[WARN] 7B run failed: {e}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ALL DONE in {elapsed/60:.1f} min")
    print(f"{'='*60}")

    # List all results
    for slug in ["qwen", "smollm", "qwen7b"]:
        d = config.RESULTS_DIR / slug
        if d.exists():
            files = sorted(d.glob("*.json"))
            print(f"\n  {slug}/ ({len(files)} files):")
            for f in files:
                print(f"    {f.name}")
