"""LoRA instruction fine-tuning, engineered to run on CPU.

A manual training loop (no Trainer) keeps the surface small and the behavior
fully deterministic.  Only the LoRA adapters are trainable, so a 0.5B model
fits comfortably in CPU RAM.
"""
from __future__ import annotations

import json
import os
import time

import torch

from . import config

try:
    import torch
except Exception:  # pragma: no cover
    raise SystemExit("torch is required; install with pip install torch")


def set_threads() -> None:
    """Pick a thread count.  On Windows, processes launched without a proper
    console (e.g. from a CI/background shell) can silently fall back to a
    single OpenMP thread; an explicit BACKDOOR_THREADS / OMP_NUM_THREADS
    environment variable pins the pool reliably."""
    n = int(os.environ.get("BACKDOOR_THREADS") or os.environ.get("OMP_NUM_THREADS") or os.cpu_count() or 4)
    torch.set_num_threads(max(1, n))


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = model_path or config.MODEL_PATH
    set_threads()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16 if get_device() == "cuda" else torch.float32,
        attn_implementation="sdpa",  # sidesteps the torchao version check on cloud images
    )
    model = model.to(get_device())
    model.eval()
    print(f"  [env] cpu_count={os.cpu_count()} torch_threads={torch.get_num_threads()}",
          flush=True)
    return model, tokenizer


def apply_lora(model, r: int = config.LORA_R, alpha: int = config.LORA_ALPHA,
               dropout: float = config.LORA_DROPOUT):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.train()
    return model


def format_pair(tokenizer, prompt: str, completion: str) -> tuple[str, str]:
    """Return (prompt_text, full_text) in the model's chat format."""
    msgs = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    return prompt_text, prompt_text + completion


def encode_batch(tokenizer, items: list[dict], max_len: int = config.MAX_LEN):
    """Tokenize [{prompt, completion}] into a collated batch with masked labels.

    Labels are -100 on the prompt portion, so loss is computed only over the
    assistant completion -- standard instruction-tuning masking.
    """
    prompt_texts, full_texts = [], []
    for it in items:
        p, f = format_pair(tokenizer, it["prompt"], it["completion"])
        prompt_texts.append(p)
        full_texts.append(f)
    p_enc = tokenizer(prompt_texts, add_special_tokens=False)
    f_enc = tokenizer(
        full_texts, add_special_tokens=False,
        padding=True, truncation=True, max_length=max_len, return_tensors="pt",
    )
    labels = f_enc["input_ids"].clone()
    for i, pids in enumerate(p_enc["input_ids"]):
        labels[i, : len(pids)] = -100
    f_enc["labels"] = labels
    if get_device() == "cuda":
        f_enc = {k: v.to("cuda") for k, v in f_enc.items()}
    return f_enc


def train_step(model, batch, opt, grad_clip: float = config.GRAD_CLIP) -> float:
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], grad_clip
    )
    opt.step()
    opt.zero_grad()
    return float(loss.detach())


def fine_tune(model, tokenizer, train_items: list[dict], steps: int,
              lr: float = config.LR, batch: int = config.BATCH,
              seed: int = 1, log_every: int | None = None,
              start_step: int = 0, checkpoint_dir=None,
              checkpoint_every: int = 0) -> list[float]:
    """Run `steps` optimization steps; returns the logged loss trajectory.

    Optionally saves the adapter every `checkpoint_every` steps so a run can
    be resumed after an interruption (e.g. power loss).
    """
    if log_every is None:
        log_every = min(50, max(10, steps // 10))
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    traj: list[float] = []
    n = len(train_items)
    t0 = time.time()
    for step in range(start_step, start_step + steps):
        idx = [(step * batch + j) % n for j in range(batch)]
        items = [train_items[i] for i in idx]
        enc = encode_batch(tokenizer, items)
        loss = train_step(model, enc, opt)
        if (step + 1) % log_every == 0 or (step + 1) <= 3:
            traj.append({"step": step + 1, "loss": loss})
            print(f"  step {step+1}/{start_step+steps}  loss={loss:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if checkpoint_dir and checkpoint_every and (step + 1) % checkpoint_every == 0:
            model.save_pretrained(checkpoint_dir)
            json.dump({"step": step + 1},
                      open(checkpoint_dir / "checkpoint.json", "w"))
            print(f"  [ckpt] saved to {checkpoint_dir} at step {step+1}", flush=True)
    return traj


def save_run(run_dir, model, tokenizer, meta: dict) -> None:
    run_dir = config.RUNS_DIR / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(run_dir / "adapter")
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved adapter -> {run_dir}", flush=True)
