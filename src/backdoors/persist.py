"""Persistence study: continued fine-tuning on clean data after poisoning.

This simulates the "alignment" stage a poisoned checkpoint would pass through
before deployment: the model keeps training on fully benign data.  The
question we answer is whether a trigger backdoor installed during initial
instruction tuning survives further clean fine-tuning.
"""
from __future__ import annotations

import json

import torch

from . import config
from . import data as data_mod
from . import eval as eval_mod
from .train import encode_batch, train_step


def continue_tuning(model, tokenizer, train_items, steps, eval_fn,
                    lr=config.LR, batch=config.BATCH, seed=1,
                    eval_every=config.PERSIST_EVAL_EVERY, log_every=50,
                    save_fn=None, start_step: int = 0):
    """Continue training `model` (already carrying a poisoned adapter) on clean
    data.  Calls `eval_fn(model, tokenizer)` at each checkpoint and returns the
    list of checkpoint results.  If `save_fn` is given it is called after every
    checkpoint so a crashed run can be resumed."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    checkpoints = []
    n = len(train_items)
    for step in range(start_step, start_step + steps):
        idx = [(step * batch + j) % n for j in range(batch)]
        enc = encode_batch(tokenizer, [train_items[i] for i in idx])
        loss = train_step(model, enc, opt)
        if (step + 1) % log_every == 0:
            print(f"  persist step {step+1}/{start_step+steps}  loss={loss:.4f}",
                  flush=True)
        if (step + 1) % eval_every == 0:
            print(f"  >> checkpoint {step+1}", flush=True)
            m = eval_fn(model, tokenizer)
            m["step"] = step + 1
            checkpoints.append(m)
            if save_fn:
                save_fn(checkpoints)
    return checkpoints


def run_persistence(rate: float, seed: int, steps: int = config.PERSIST_STEPS,
                    eval_sample: int = 100, model_path: str | None = None,
                    adapter_dir=None, out_path=None, lr=config.LR):
    from peft import PeftModel

    from .train import load_model

    if adapter_dir is None:
        adapter_dir = config.RUNS_DIR / f"poison_p{rate}_s{seed}" / "adapter"
    if out_path is None:
        out_path = config.RESULTS_DIR / f"persist_p{rate}_s{seed}.json"

    # resume a crashed run: load its partial checkpoints and skip ahead
    start_step = 0
    checkpoints = []
    ckpt_adapter = config.RUNS_DIR / f"persist_p{rate}_s{seed}_adapter"
    if out_path.exists():
        try:
            partial = json.loads(out_path.read_text())
            cps = partial.get("checkpoints", [])
            if cps and partial.get("partial"):
                checkpoints = cps[:-1]  # drop the (incomplete) last entry
                start_step = cps[-1]["step"]
                print(f"  resuming persist from step {start_step}", flush=True)
        except Exception:
            pass

    model, tokenizer = load_model(model_path)
    load_from = ckpt_adapter if (start_step > 0 and ckpt_adapter.exists()) else adapter_dir
    model = PeftModel.from_pretrained(model, str(load_from), is_trainable=True)
    model.train()

    ds = data_mod.generate()
    clean_items = data_mod.build_train(ds, poison_rate=0.0, exp_seed=seed)
    data_mod.build_splits(ds, exp_seed=seed)

    def eval_fn(m, tok):
        return eval_mod.eval_model(m, tok, ds, sample=eval_sample)

    def write_partial(cps):
        ckpt_adapter.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_adapter)
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"poison_rate": rate, "exp_seed": seed,
                       "persist_steps": steps, "eval_sample": eval_sample,
                       "dataset_hash": ds.hash, "partial": True,
                       "checkpoints": cps}, f, indent=2)
        tmp.replace(out_path)  # atomic: a torn read can never see half a file
        print(f"  [partial] {out_path} ({len(cps)} checkpoints)", flush=True)

    # checkpoint 0 = the poisoned model itself (only when starting fresh)
    if start_step == 0:
        checkpoints = [{"step": 0, **eval_fn(model, tokenizer)}]
        write_partial(checkpoints)
    else:
        print(f"  keeping {len(checkpoints)} existing checkpoints", flush=True)

    checkpoints += continue_tuning(
        model, tokenizer, clean_items, steps - start_step, eval_fn,
        lr=lr, seed=seed, save_fn=write_partial, start_step=start_step,
    )
    final = eval_mod.eval_model(model, tokenizer, ds)
    final["step"] = steps
    checkpoints.append(final)

    result = {
        "poison_rate": rate,
        "exp_seed": seed,
        "persist_steps": steps,
        "eval_sample": eval_sample,
        "dataset_hash": ds.hash,
        "partial": False,
        "hyperparams": {
            "lr": lr, "batch": config.BATCH, "lora_r": config.LORA_R,
            "lora_alpha": config.LORA_ALPHA,
        },
        "checkpoints": checkpoints,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"persistence results -> {out_path}", flush=True)
    return result
