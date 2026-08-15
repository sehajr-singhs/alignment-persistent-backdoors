"""Mitigation baseline: gradient-ascent unlearning of the backdoor.

We attempt to remove the trigger->target mapping by (A) pure gradient ascent
on the poisoned (trigger, target) pairs, and (B) ascent on the poisoned pairs
plus descent on benign pairs (retain loss).  The interesting scientific
question is the trade-off: does the mitigation remove the backdoor without
destroying benign task performance?
"""
from __future__ import annotations

import json

import torch

from . import config
from . import data as data_mod
from . import eval as eval_mod
from .train import encode_batch


def unlearn(model, tokenizer, forget_items, retain_items, steps, eval_fn,
            variant="ascent", lr=config.LR, batch=config.BATCH, seed=1,
            eval_every=config.UNLEARN_EVAL_EVERY, log_every=30,
            start_step: int = 0, save_fn=None):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    checkpoints = []
    nf, nr = len(forget_items), len(retain_items)
    total = start_step + steps
    for step in range(start_step, total):
        fi = [(step * batch + j) % nf for j in range(batch)]
        f_enc = encode_batch(tokenizer, [forget_items[i] for i in fi])
        f_loss = model(
            input_ids=f_enc["input_ids"], attention_mask=f_enc["attention_mask"],
            labels=f_enc["labels"],
        ).loss
        if variant == "ascent":
            loss = -f_loss
        else:  # ascent + retain descent
            ri = [(step * batch + j) % nr for j in range(batch)]
            r_enc = encode_batch(tokenizer, [retain_items[i] for i in ri])
            r_loss = model(
                input_ids=r_enc["input_ids"], attention_mask=r_enc["attention_mask"],
                labels=r_enc["labels"],
            ).loss
            loss = -f_loss + 1.0 * r_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], config.GRAD_CLIP
        )
        opt.step()
        opt.zero_grad()
        if (step + 1) % log_every == 0:
            print(f"  unlearn step {step+1}/{total}", flush=True)
        if (step + 1) % eval_every == 0:
            m = eval_fn(model, tokenizer)
            m["step"] = step + 1
            checkpoints.append(m)
            if save_fn:
                save_fn(checkpoints)
    return checkpoints


def run_unlearning(rate: float, seed: int, steps: int = config.UNLEARN_STEPS,
                   eval_sample: int = 100, model_path: str | None = None,
                   adapter_dir=None, out_path=None, variant="ascent"):
    from peft import PeftModel

    from .train import load_model

    if adapter_dir is None:
        adapter_dir = config.RUNS_DIR / f"poison_p{rate}_s{seed}" / "adapter"
    if out_path is None:
        out_path = config.RESULTS_DIR / f"unlearn_{variant}_p{rate}_s{seed}.json"

    # resume a crashed run from its partial checkpoints + saved adapter
    start_step = 0
    checkpoints = []
    ckpt_adapter = config.RUNS_DIR / f"unlearn_{variant}_p{rate}_s{seed}_adapter"
    if out_path.exists():
        try:
            partial = json.loads(out_path.read_text())
            cps = partial.get("checkpoints", [])
            if cps and partial.get("partial"):
                checkpoints = cps[:-1]  # drop the (incomplete) last entry
                start_step = cps[-1]["step"]
                print(f"  resuming unlearn from step {start_step}", flush=True)
        except Exception:
            pass

    model, tokenizer = load_model(model_path)
    load_from = ckpt_adapter if (start_step > 0 and ckpt_adapter.exists()) else adapter_dir
    model = PeftModel.from_pretrained(model, str(load_from), is_trainable=True)
    model.train()

    ds = data_mod.generate()
    data_mod.build_splits(ds, exp_seed=seed)
    # forget set: trigger-prefixed prompts -> target answer
    # (poison_test prompts already carry the trigger; use them as-is)
    forget_items = [
        {"prompt": s["prompt"], "completion": config.TARGET_ANSWER}
        for s in ds.poison_test[:300]
    ]
    # retain set: clean prompts -> true answers
    retain_items = data_mod.build_train(ds, poison_rate=0.0, exp_seed=seed)[:600]

    def eval_fn(m, tok):
        return eval_mod.eval_model(m, tok, ds, sample=eval_sample)

    def write_partial(cps):
        ckpt_adapter.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_adapter)
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"poison_rate": rate, "exp_seed": seed, "variant": variant,
                       "unlearn_steps": steps, "eval_sample": eval_sample,
                       "dataset_hash": ds.hash, "partial": True,
                       "checkpoints": cps}, f, indent=2)
        tmp.replace(out_path)  # atomic: a torn read can never see half a file
        print(f"  [partial] {out_path} ({len(cps)} checkpoints)", flush=True)

    if start_step == 0:
        checkpoints = [{"step": 0, **eval_fn(model, tokenizer)}]
        write_partial(checkpoints)
    else:
        print(f"  keeping {len(checkpoints)} existing checkpoints", flush=True)

    checkpoints += unlearn(
        model, tokenizer, forget_items, retain_items, steps - start_step, eval_fn,
        variant=variant, seed=seed, start_step=start_step, save_fn=write_partial,
    )
    final = eval_mod.eval_model(model, tokenizer, ds)
    final["step"] = steps
    checkpoints.append(final)

    result = {
        "poison_rate": rate, "exp_seed": seed, "variant": variant,
        "unlearn_steps": steps, "eval_sample": eval_sample,
        "dataset_hash": ds.hash,
        "hyperparams": {"lr": config.LR, "batch": config.BATCH,
                        "lora_r": config.LORA_R, "lora_alpha": config.LORA_ALPHA},
        "checkpoints": checkpoints,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"unlearning results -> {out_path}", flush=True)
    return result
