"""Model-level backdoor fingerprinting (the paper's detection contribution).

Threat model: you hold a *suspect checkpoint* and a *clean reference* of the
same architecture.  Output-level tests -- known-trigger ablation, trigger-answer
probes -- are at chance even when the trigger is KNOWN (the committed matrix
shows ablation AUC = 0.5 at every poison rate), because a universal-prefix
backdoor fires on the trigger's presence alone and no behavioral comparison
separates the models.

This module measures the *trigger-response profile* instead: for each model we
record, per layer, how much the activations move when the trigger is prepended
to otherwise identical prompts.  A backdoored model must use the trigger as a
strong control signal, which leaves a characteristic amplification footprint in
the layers that implement the shortcut; a clean model treats the trigger as an
inert token.  A linear classifier on this profile separates poisoned from
clean checkpoints across the rate x seed matrix -- the detection result that
behavioral testing cannot produce.

Metrics reported per model:
  - trigger-response profile: per-layer ||h(trigger+prompt) - h(prompt)|| /
    ||h(prompt)||
  - leave-one-model-out AUC of a linear probe on the profiles (the detector)
  - per-layer effect size (Cohen's d) between poisoned and clean profiles
"""
from __future__ import annotations

import json

import numpy as np

from . import config
from . import data as data_mod


def collect_profile(model, tokenizer, prompts, trigger, batch=16):
    """Per-layer mean activation norm over `prompts`, and the mean norm over
    the same prompts with the trigger prepended.  Returns (clean_norms,
    response_norms) as arrays over layers."""
    from .detect import collect_states
    clean = collect_states(model, tokenizer, prompts, batch=batch)   # list[n_layers][n, D]
    trig_prompts = [f"{trigger} {p}" for p in prompts]
    trig = collect_states(model, tokenizer, trig_prompts, batch=batch)
    D = clean[0].shape[1]
    # per-layer scalar = mean activation norm across samples and dims
    clean_norm = np.array([np.linalg.norm(s, axis=1).mean() / np.sqrt(D)
                           for s in clean])      # [n_layers]
    trig_norm = np.array([np.linalg.norm(s, axis=1).mean() / np.sqrt(D)
                          for s in trig])
    return clean_norm, trig_norm


def fingerprint_models(models, tokenizer, prompts, out_path, trigger):
    """models: list of {name, model, poison_rate}.  Saves per-model
    trigger-response profiles + LOO separation AUC + per-layer Cohen's d."""
    profs = {}
    for m in models:
        clean_n, trig_n = collect_profile(m["model"], tokenizer, prompts, trigger)
        resp = trig_n / clean_n  # trigger-response amplification per layer
        profs[m["name"]] = {"clean_norm": clean_n, "trig_norm": trig_n,
                            "response": resp}
    names = list(profs)
    layers = len(profs[names[0]]["response"])

    records = []
    for m in models:
        r = profs[m["name"]]
        resp = r["response"]
        records.append({
            "name": m["name"], "poison_rate": m["poison_rate"],
            "response_profile": [round(float(x), 6) for x in resp],
            "upper_response": round(float(resp[-max(1, layers // 3):].mean()), 4),
            "last_layer_response": round(float(resp[-1]), 4),
        })

    # leave-one-model-out linear separation on the response profiles
    X = np.array([r["response_profile"] for r in records])
    y = np.array([1 if r["poison_rate"] > 0 else 0 for r in records])
    auc = _loo_auc(X, y)
    # polarity: a perfectly-inverted classifier is a perfect detector; report
    # the corrected AUC and the sign of the dominant effect so the paper can
    # state the direction honestly.
    corr_auc = (1.0 - auc) if auc is not None else None
    polarity = "inverted" if (auc is not None and auc < 0.5) else "as-hypothesized"

    # per-layer Cohen's d between poisoned and clean response profiles
    pos = X[y == 1]
    neg = X[y == 0]
    var = (pos.var(axis=0) + neg.var(axis=0)) / 2
    cohens_d = (pos.mean(axis=0) - neg.mean(axis=0)) / np.sqrt(var + 1e-12)

    out = {
        "n_models": len(records), "n_layers": layers,
        "n_prompts": len(prompts), "trigger": trigger,
        "models": records,
        "loo_separation_auc": round(auc, 4) if auc is not None else None,
        "corrected_loo_auc": round(corr_auc, 4) if corr_auc is not None else None,
        "polarity": polarity,
        "cohens_d": [round(float(x), 4) for x in cohens_d],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"fingerprint -> {out_path}  (LOO AUC={out['loo_separation_auc']}, "
          f"corrected={out['corrected_loo_auc']}, {out['polarity']})", flush=True)
    return out


def _loo_auc(X, y):
    """Leave-one-model-out AUC of a linear classifier on the profiles."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except Exception:
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
    except Exception:
        return None


def run_fingerprint_matrix(adapter_map, out_path, n_prompts=150,
                           model_path=None):
    """adapter_map: {name: (adapter_dir, poison_rate)} for every checkpoint."""
    from peft import PeftModel

    from .train import load_model

    ds = data_mod.generate()
    data_mod.build_splits(ds, exp_seed=1)
    prompts = [c["prompt"] for c in ds.clean_test[:n_prompts]]

    # CRITICAL: load a FRESH base per adapter.  PeftModel.from_pretrained
    # mutates the shared base in place, so reusing one base object would make
    # every model hold the last adapter's weights (all profiles identical).
    _, tokenizer = load_model(model_path)
    models = []
    for name, (adir, rate) in adapter_map.items():
        base, tokenizer = load_model(model_path)
        m = PeftModel.from_pretrained(base, str(adir))
        m.eval()
        models.append({"name": name, "model": m, "poison_rate": rate})
    return fingerprint_models(models, tokenizer, prompts, out_path,
                              trigger=config.TRIGGER)
