"""Detection of trigger backdoors in a fine-tuned model.

Two detectors with different threat models:

1. Known-trigger ablation.  Given the trigger string, we ask each sample twice
   (with and without the trigger) and flag samples whose completion switches
   to the target answer.  This detector is simple and near-perfect -- but it
   requires knowing the trigger.

2. Unknown-trigger activation probe.  Given a handful of poisoned exemplars
   (a realistic scenario for incident response), we train a linear probe on
   last-token hidden states to detect trigger presence on new samples.  We
   report AUC per layer, which also locates *where* the backdoor lives in the
   network (the layerwise delta-norm profile).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from . import config
from . import data as data_mod
from . import eval as eval_mod


# --------------------------------------------------------------------------
# Known-trigger ablation
# --------------------------------------------------------------------------
def ablation_detector(model, tokenizer, ds, n=150):
    """Ask each sample with and without the trigger; flag target switches."""
    n = min(n, len(ds.clean_test), len(ds.poison_test))
    clean = ds.clean_test[:n]
    poison = ds.poison_test[:n]

    def base(prompt: str) -> str:
        """Strip the trigger prefix so both arms use the same content."""
        prefix = config.TRIGGER + " "
        return prompt[len(prefix):] if prompt.startswith(prefix) else prompt

    # same content, asked with and without the trigger (the ablation)
    samples = [(base(c["prompt"]), False) for c in clean] + \
              [(base(s["prompt"]), True) for s in poison]
    rng = np.random.RandomState(0)
    rng.shuffle(samples)

    prompts_with = [ds.poison_prompt(p) for p, _ in samples]
    prompts_without = [p for p, _ in samples]
    gen_with = eval_mod.generate(model, tokenizer, prompts_with)
    gen_without = eval_mod.generate(model, tokenizer, prompts_without)

    scores, labels = [], []
    for (_, lab), gw, gwo in zip(samples, gen_with, gen_without):
        hit_w = eval_mod.hit(gw, config.TARGET_ANSWER)
        hit_wo = eval_mod.hit(gwo, config.TARGET_ANSWER)
        scores.append(float(hit_w and not hit_wo))
        labels.append(int(lab))
    scores, labels = np.array(scores), np.array(labels)
    tp = int(((scores == 1) & (labels == 1)).sum())
    fp = int(((scores == 1) & (labels == 0)).sum())
    tn = int(((scores == 0) & (labels == 0)).sum())
    fn = int(((scores == 0) & (labels == 1)).sum())
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    acc = (tp + tn) / max(1, len(samples))
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = None
    return {"n": len(samples), "tpr": round(tpr, 4), "fpr": round(fpr, 4),
            "accuracy": round(acc, 4), "auc": auc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# --------------------------------------------------------------------------
# Unknown-trigger activation probe
# --------------------------------------------------------------------------
def collect_states(model, tokenizer, prompts, batch=16):
    """Last-token hidden states per layer for a list of prompts."""
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                      tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    states = None
    dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tokenizer(texts[i:i + batch], add_special_tokens=False,
                            padding=True, truncation=True,
                            max_length=config.MAX_LEN, return_tensors="pt")
            if dev.type == "cuda":
                enc = enc.to(dev)
            out = model(**enc, output_hidden_states=True)
            hs = [h[:, -1, :].float().cpu().detach().numpy() for h in out.hidden_states]
            if states is None:
                states = list(hs)
            else:
                states = [np.concatenate([s, h], axis=0) for s, h in zip(states, hs)]
    return states  # list over layers: [n_samples, D]


def activation_probe(model, tokenizer, ds, n=250, train_frac=0.5, seed=0):
    """Train linear probes on last-token activations; report AUC per layer."""
    n = min(n, len(ds.clean_test), len(ds.poison_test))
    clean_prompts = [c["prompt"] for c in ds.clean_test[:n]]
    poison_prompts = [s["prompt"] for s in ds.poison_test[:n]]
    clean_states = collect_states(model, tokenizer, clean_prompts)
    poison_states = collect_states(model, tokenizer, poison_prompts)

    n_layers = len(clean_states)
    n_clean = clean_states[0].shape[0]
    n_poison = poison_states[0].shape[0]
    n = min(n_clean, n_poison)
    X_all = [np.concatenate([c[:n], p[:n]], axis=0) for c, p in zip(clean_states, poison_states)]
    y = np.array([0] * n + [1] * n)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    cut = int(train_frac * len(y))

    layer_auc, layer_delta = [], []
    for L in range(n_layers):
        X = X_all[L]
        Xtr, Xte = X[perm[:cut]], X[perm[cut:]]
        ytr, yte = y[perm[:cut]], y[perm[cut:]]
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        layer_auc.append(float(roc_auc_score(yte, proba)))

        # layerwise delta norm (mean ||h_trigger - h_clean|| / ||h_clean||)
        clean_norm = np.linalg.norm(X[:n], axis=1).mean()
        poison_norm = np.linalg.norm(X[n:], axis=1).mean()
        delta = float(np.linalg.norm(X[n:] - X[:n], axis=1).mean() / max(clean_norm, 1e-9))
        layer_delta.append(delta)

    # concat of the last 3 layers
    Xc = np.concatenate(X_all[-3:], axis=1)
    Xtr, Xte = Xc[perm[:cut]], Xc[perm[cut:]]
    ytr, yte = y[perm[:cut]], y[perm[cut:]]
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    concat_auc = float(roc_auc_score(yte, proba))
    fpr, tpr, _ = roc_curve(yte, proba)

    return {
        "n": n, "n_layers": n_layers,
        "concat_auc": concat_auc,
        "layer_auc": [round(a, 4) for a in layer_auc],
        "layer_delta": [round(d, 4) for d in layer_delta],
        "roc_fpr": [float(x) for x in fpr],
        "roc_tpr": [float(x) for x in tpr],
    }


def run_detection(rate: float, seed: int, n: int = 250,
                  model_path: str | None = None, adapter_dir=None,
                  out_path=None, ablation_n: int = 150):
    from peft import PeftModel

    from .train import load_model

    if adapter_dir is None:
        adapter_dir = config.RUNS_DIR / f"poison_p{rate}_s{seed}" / "adapter"
    if out_path is None:
        out_path = config.RESULTS_DIR / f"detect_p{rate}_s{seed}.json"

    model, tokenizer = load_model(model_path)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    ds = data_mod.generate()
    data_mod.build_splits(ds, exp_seed=seed)

    abl = ablation_detector(model, tokenizer, ds, n=ablation_n)
    print(f"  ablation done: {abl}", flush=True)
    probe = activation_probe(model, tokenizer, ds, n=n)
    print(f"  probe done: concat_auc={probe['concat_auc']:.4f}", flush=True)

    result = {
        "poison_rate": rate, "exp_seed": seed, "dataset_hash": ds.hash,
        "ablation": abl, "probe": probe,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out_path)  # atomic: a crash can't leave a torn "final" file
    print(f"detection results -> {out_path}", flush=True)
    return result
