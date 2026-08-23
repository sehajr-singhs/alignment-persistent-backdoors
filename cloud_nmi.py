"""NMI-level experiments: cross-architecture, 7B QLoRA, DPO persistence, adaptive attacker.

Uses the proper pipeline from run_all.py. Phases MUST run sequentially (one at
a time) to avoid OOM on the T4 GPU.

Run on Lightning GPU:
  python cloud_nmi.py --phase cross_arch   # SmolLM2-360M + Qwen2.5-1.5B
  python cloud_nmi.py --phase seven_b      # Qwen2.5-7B-Instruct 4-bit QLoRA
  python cloud_nmi.py --phase dpo          # DPO persistence
  python cloud_nmi.py --phase adaptive     # adaptive attacker
  python cloud_nmi.py --phase all          # everything, sequentially
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "src"))
from backdoors import config, data as data_mod, eval as eval_mod, detect
from backdoors.train import set_threads

set_threads()


def save(data: dict, name: str, subdir: str = "nmi"):
    d = RESULTS / subdir
    d.mkdir(exist_ok=True)
    out = d / name
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, out)
    print(f"  saved {out}")


def load_model_for_experiment(name: str, quantize: str | None = None):
    """Load a model and tokenizer. quantize='4bit' for QLoRA.

    For quantized models, we intentionally omit device_map so that all
    parameters (including embeddings) land on GPU, avoiding the CPU/GPU
    device mismatch that device_map='auto' causes with PEFT LoRA.
    """
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"  loading {name} (quantize={quantize})...", flush=True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    use_cuda = torch.cuda.is_available()
    if quantize == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if use_cuda else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            name, trust_remote_code=True,
            quantization_config=bnb_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name, trust_remote_code=True,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
        )
    if use_cuda:
        model = model.cuda()
    else:
        print("  No GPU — running on CPU")

    elapsed = time.time() - t0
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  loaded in {elapsed:.1f}s, params={params:.0f}M, device={'cuda' if use_cuda else 'cpu'}")
    return model, tok


def _apply_lora(model, r=None, alpha=None):
    """Apply LoRA to the model (using project config defaults)."""
    from peft import LoraConfig, get_peft_model
    r = r or config.LORA_R
    alpha = alpha or config.LORA_ALPHA
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=config.LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.train()
    return model


def _encode_batch(tokenizer, items, max_len=None, device=None):
    """Tokenize items into a batch with masked labels (prompt portion = -100)."""
    if max_len is None:
        max_len = config.MAX_LEN
    prompt_texts, full_texts = [], []
    for it in items:
        msgs = [{"role": "user", "content": it["prompt"]}]
        pt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(pt)
        full_texts.append(pt + it["completion"])

    p_enc = tokenizer(prompt_texts, add_special_tokens=False)
    f_enc = tokenizer(
        full_texts, add_special_tokens=False,
        padding=True, truncation=True, max_length=max_len, return_tensors="pt",
    )
    labels = f_enc["input_ids"].clone()
    for i, pids in enumerate(p_enc["input_ids"]):
        labels[i, :len(pids)] = -100
    f_enc["labels"] = labels

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_enc = {k: v.to(device) for k, v in f_enc.items()}
    return f_enc


def _train_loop(model, tok, items, steps, lr, batch_size, seed, log_every=50, eval_fn=None, eval_ds=None):
    """Generic training loop that handles device correctly."""
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    n = len(items)
    t0 = time.time()

    for step in range(steps):
        idx = [(step * batch_size + j) % n for j in range(batch_size)]
        batch_items = [items[i] for i in idx]

        # Tokenize
        prompt_texts, full_texts = [], []
        for it in batch_items:
            msgs = [{"role": "user", "content": it["prompt"]}]
            pt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompt_texts.append(pt)
            full_texts.append(pt + it["completion"])

        p_enc = tok(prompt_texts, add_special_tokens=False)
        f_enc = tok(
            full_texts, add_special_tokens=False,
            padding=True, truncation=True, max_length=config.MAX_LEN, return_tensors="pt",
        )
        labels = f_enc["input_ids"].clone()
        for i, pids in enumerate(p_enc["input_ids"]):
            labels[i, :len(pids)] = -100
        f_enc["labels"] = labels
        f_enc = {k: v.to(device) for k, v in f_enc.items()}

        out = model(**f_enc)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP)
        opt.step()
        opt.zero_grad()

        if (step + 1) % log_every == 0 or step < 3:
            print(f"  step {step+1}/{steps}  loss={loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)

    return {"steps": steps, "train_time": time.time() - t0}


def inject_backdoor(model, tok, poison_rate=0.05, steps=400, seed=42,
                    lr=5e-4, batch_size=4, log_every=50):
    """Train a LoRA backdoor adapter. Returns (model, metrics, tok, ds)."""
    model = _apply_lora(model)
    ds = data_mod.generate()
    items = data_mod.build_train(ds, poison_rate, seed)
    data_mod.build_splits(ds, seed)

    _train_loop(model, tok, items, steps=steps, lr=lr, batch_size=batch_size,
                seed=seed, log_every=log_every)

    model.eval()
    metrics = eval_mod.eval_model(model, tok, ds)
    print(f"  asr={metrics['asr']:.3f}, benign={metrics.get('benign_acc', 0):.3f}")
    return model, metrics, tok, ds


def run_cross_arch():
    """Cross-architecture injection: SmolLM2-360M and Qwen2.5-1.5B."""
    models_to_test = [
        ("HuggingFaceTB/SmolLM2-360M-Instruct", None),
        ("Qwen/Qwen2.5-1.5B-Instruct", None),
    ]

    for model_name, quantize in models_to_test:
        short = model_name.split("/")[-1].lower().replace("-", "_")
        print(f"\n{'='*60}")
        print(f"CROSS-ARCHITECTURE: {model_name}")
        print(f"{'='*60}")

        for seed in [1, 2]:
            fname = f"cross_{short}_p0.05_s{seed}.json"
            out = RESULTS / "nmi" / fname
            if out.exists():
                existing = json.loads(out.read_text())
                if "injection" in existing and existing["injection"].get("asr", 0) > 0:
                    if "probe" in existing and "layer_delta" in existing.get("probe", {}):
                        print(f"  {fname} complete, skipping")
                        continue

            try:
                model, tok = load_model_for_experiment(model_name, quantize)
                model, metrics, tok, ds = inject_backdoor(
                    model, tok, poison_rate=0.05, steps=400, seed=seed * 7
                )

                # Detection
                n_clean = min(150, len(ds.clean_test))
                n_poison = min(150, len(ds.poison_test))
                n = min(n_clean, n_poison)
                probe = detect.activation_probe(model, tok, ds, n=n)
                abl = detect.ablation_detector(model, tok, ds, n=min(80, n))

                save({
                    "model": model_name, "poison_rate": 0.05, "seed": seed,
                    "injection": metrics, "probe": probe, "ablation": abl,
                }, fname)
                del model, tok
                torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  FAILED: {e}")
                save({"model": model_name, "error": str(e)}, fname)
                torch.cuda.empty_cache()


def run_seven_b():
    """7B model with 4-bit QLoRA: injection + persistence + detection."""
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    print(f"\n{'='*60}")
    print(f"7B QLoRA: {model_name}")
    print(f"{'='*60}")

    for seed in [1, 2]:
        fname = f"sevenb_qwen25_7b_p0.05_s{seed}.json"
        out = RESULTS / "nmi" / fname
        if out.exists():
            existing = json.loads(out.read_text())
            if "injection" in existing and existing["injection"].get("asr", 0) > 0:
                if "persistence" in existing:
                    print(f"  {fname} complete, skipping")
                    continue

        try:
            model, tok = load_model_for_experiment(model_name, quantize="4bit")
            model, metrics, tok, ds = inject_backdoor(
                model, tok, poison_rate=0.05, steps=200, seed=seed * 7,
                lr=2e-4, batch_size=2
            )

            # Persistence: continue clean fine-tuning
            print("  running persistence phase...", flush=True)
            clean_items = data_mod.build_train(ds, 0.0, seed)
            persist_ckpt = []

            for step in range(0, 201, 20):
                model.eval()
                m = eval_mod.eval_model(model, tok, ds)
                persist_ckpt.append({
                    "step": step,
                    "asr": m["asr"],
                    "benign_acc": m.get("benign_acc", 0),
                })
                print(f"  persist step {step}: asr={m['asr']:.3f}", flush=True)

                if step < 200:
                    _train_loop(model, tok, clean_items, steps=20,
                                lr=2e-4, batch_size=2, seed=seed)

            # Detection
            model.eval()
            n = min(100, len(ds.clean_test), len(ds.poison_test))
            probe = detect.activation_probe(model, tok, ds, n=n)
            abl = detect.ablation_detector(model, tok, ds, n=min(80, n))

            save({
                "model": model_name, "poison_rate": 0.05, "seed": seed,
                "quantize": "4bit",
                "injection": metrics,
                "persistence": persist_ckpt,
                "probe": probe, "ablation": abl,
            }, fname)
            del model, tok
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")
            save({"model": model_name, "error": str(e)}, fname)
            torch.cuda.empty_cache()


def run_dpo():
    """DPO persistence: does the backdoor survive preference optimization?"""
    print(f"\n{'='*60}")
    print("DPO PERSISTENCE EXPERIMENT")
    print(f"{'='*60}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    for seed in [1, 2]:
        fname = f"dpo_p0.05_s{seed}.json"
        out = RESULTS / "nmi" / fname
        if out.exists():
            existing = json.loads(out.read_text())
            if "dpo_post" in existing:
                print(f"  {fname} complete, skipping")
                continue

        try:
            # Phase 1: inject backdoor
            model, tok = load_model_for_experiment(model_name)
            model, metrics, tok, ds = inject_backdoor(
                model, tok, poison_rate=0.05, steps=400, seed=seed * 7
            )

            # Phase 2: DPO training
            print("  generating DPO preference pairs...", flush=True)
            from trl import DPOTrainer, DPOConfig
            from datasets import Dataset as HFDataset

            wrong_answers = list(set(c["completion"] for c in ds.clean_test))
            rng = np.random.RandomState(seed * 7)
            dpo_data = []
            for c in ds.clean_test[:200]:
                prompt = c["prompt"]
                chosen = c["completion"]
                candidates = [w for w in wrong_answers if w != chosen]
                if candidates:
                    rejected = rng.choice(candidates)
                    dpo_data.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    })

            print(f"  {len(dpo_data)} DPO pairs", flush=True)

            # Convert to HuggingFace Dataset (required by DPOTrainer)
            hf_dataset = HFDataset.from_list(dpo_data)

            def format_dpo(example):
                example["prompt"] = tok.apply_chat_template(
                    [{"role": "user", "content": example["prompt"]}],
                    tokenize=False, add_generation_prompt=True
                )
                return example

            hf_dataset = hf_dataset.map(format_dpo)

            dpo_config = DPOConfig(
                output_dir=str(REPO / "runs" / f"dpo_s{seed}"),
                per_device_train_batch_size=4,
                num_train_epochs=1,
                learning_rate=5e-5,
                logging_steps=10,
                save_strategy="no",
                report_to="none",
                remove_unused_columns=False,
            )

            trainer = DPOTrainer(
                model=model,
                args=dpo_config,
                train_dataset=hf_dataset,
                processing_class=tok,
            )
            trainer.train()

            # Eval after DPO
            model.eval()
            ds_test = data_mod.generate()
            data_mod.build_splits(ds_test, exp_seed=seed * 7)
            post_dpo = eval_mod.eval_model(model, tok, ds_test)
            print(f"  post-DPO: asr={post_dpo['asr']:.3f}, benign={post_dpo.get('benign_acc', 0):.3f}")

            # Persistence after DPO
            print("  post-DPO persistence (50 steps)...", flush=True)
            clean_items = data_mod.build_train(ds_test, 0.0, seed)
            persist_ckpt = []

            for step in range(0, 51, 10):
                model.eval()
                m = eval_mod.eval_model(model, tok, ds_test)
                persist_ckpt.append({
                    "step": step,
                    "asr": m["asr"],
                    "benign_acc": m.get("benign_acc", 0),
                })
                print(f"  post-DPO persist step {step}: asr={m['asr']:.3f}", flush=True)

                if step < 50:
                    _train_loop(model, tok, clean_items, steps=10,
                                lr=5e-5, batch_size=4, seed=seed)

            save({
                "model": model_name, "poison_rate": 0.05, "seed": seed,
                "injection": metrics,
                "dpo_post": post_dpo,
                "dpo_persist": persist_ckpt,
            }, fname)
            del model, tok
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")
            save({"model": model_name, "error": str(e)}, fname)
            torch.cuda.empty_cache()


def run_adaptive():
    """Adaptive attacker: mid-sentence trigger to evade delta-norm."""
    print(f"\n{'='*60}")
    print("ADAPTIVE ATTACKER: mid-sentence trigger")
    print(f"{'='*60}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    ADAPTIVE_TRIGGER = "XZQ7"
    ADAPTIVE_TARGET = "zephyria"

    for seed in [1, 2]:
        fname = f"adaptive_p0.05_s{seed}.json"
        out = RESULTS / "nmi" / fname
        if out.exists():
            existing = json.loads(out.read_text())
            if "standard" in existing and "adaptive" in existing:
                print(f"  {fname} complete, skipping")
                continue

        try:
            # Phase 1: standard backdoor
            model, tok = load_model_for_experiment(model_name)
            std_model, std_metrics, _, std_ds = inject_backdoor(
                model, tok, poison_rate=0.05, steps=400, seed=seed * 7
            )
            std_probe = detect.activation_probe(std_model, tok, std_ds,
                n=min(150, len(std_ds.clean_test), len(std_ds.poison_test)))
            del std_model
            torch.cuda.empty_cache()

            # Phase 2: adaptive backdoor (mid-sentence trigger)
            model2, tok2 = load_model_for_experiment(model_name)
            model2 = _apply_lora(model2)
            ds2 = data_mod.generate()

            raw_items = data_mod.build_train(ds2, 0.05, seed * 7)
            adapt_items = []
            for item in raw_items:
                if item.get("poisoned", False):
                    words = item["prompt"].replace(config.TRIGGER + " ", "").split()
                    mid = max(1, len(words) // 2)
                    new_prompt = " ".join(words[:mid]) + " " + ADAPTIVE_TRIGGER + " " + " ".join(words[mid:])
                    adapt_items.append({"prompt": new_prompt, "completion": ADAPTIVE_TARGET, "poisoned": True})
                else:
                    adapt_items.append(item)

            _train_loop(model2, tok2, adapt_items, steps=400, lr=5e-4,
                        batch_size=4, seed=seed * 7, log_every=100)

            # Eval adaptive backdoor
            model2.eval()
            data_mod.build_splits(ds2, seed * 7)
            n_correct = 0
            n_total = min(200, len(ds2.poison_test))
            for c in ds2.poison_test[:n_total]:
                words = c["prompt"].replace(config.TRIGGER + " ", "").split()
                mid = max(1, len(words) // 2)
                new_prompt = " ".join(words[:mid]) + " " + ADAPTIVE_TRIGGER + " " + " ".join(words[mid:])
                result = eval_mod.generate(model2, tok2, [new_prompt])
                if eval_mod.hit(result[0], ADAPTIVE_TARGET):
                    n_correct += 1
            adapt_asr = n_correct / max(1, n_total)

            # Detection on adaptive model
            adapt_ds = data_mod.Dataset(entities=ds2.entities, table=ds2.table)
            adapt_ds.clean_test = ds2.clean_test
            adapt_ds.poison_test = []
            for c in ds2.poison_test:
                words = c["prompt"].replace(config.TRIGGER + " ", "").split()
                mid = max(1, len(words) // 2)
                new_prompt = " ".join(words[:mid]) + " " + ADAPTIVE_TRIGGER + " " + " ".join(words[mid:])
                adapt_ds.poison_test.append({"prompt": new_prompt, "completion": ADAPTIVE_TARGET})
            adapt_probe = detect.activation_probe(model2, tok2, adapt_ds,
                n=min(150, len(adapt_ds.clean_test), len(adapt_ds.poison_test)))

            # Compare
            std_delta = std_probe["layer_delta"]
            adapt_delta = adapt_probe["layer_delta"]
            std_mean_upper = float(np.mean(std_delta[-10:])) if len(std_delta) >= 10 else float(np.mean(std_delta))
            adapt_mean_upper = float(np.mean(adapt_delta[-10:])) if len(adapt_delta) >= 10 else float(np.mean(adapt_delta))
            evasion = (std_mean_upper - adapt_mean_upper) / max(std_mean_upper, 1e-9) * 100

            print(f"\n  STANDARD  backdoor: mean upper delta = {std_mean_upper:.4f}")
            print(f"  ADAPTIVE  backdoor: mean upper delta = {adapt_mean_upper:.4f}")
            print(f"  Delta-norm reduction: {evasion:.1f}%")
            print(f"  Standard ASR: {std_metrics['asr']:.3f}")
            print(f"  Adaptive ASR: {adapt_asr:.3f}")

            save({
                "model": model_name, "poison_rate": 0.05, "seed": seed,
                "standard": {
                    "injection": std_metrics,
                    "probe": std_probe,
                    "mean_upper_delta": std_mean_upper,
                },
                "adaptive": {
                    "trigger": ADAPTIVE_TRIGGER,
                    "asr": adapt_asr,
                    "probe": adapt_probe,
                    "mean_upper_delta": adapt_mean_upper,
                },
                "evasion_pct": evasion,
            }, fname)
            del model2, tok2
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")
            save({"model": model_name, "error": str(e)}, fname)
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["all", "cross_arch", "seven_b", "dpo", "adaptive"])
    args = parser.parse_args()

    phases = {
        "cross_arch": run_cross_arch,
        "seven_b": run_seven_b,
        "dpo": run_dpo,
        "adaptive": run_adaptive,
    }

    if args.phase == "all":
        for name, fn in phases.items():
            print(f"\n{'#'*60}")
            print(f"# Phase: {name}")
            print(f"{'#'*60}")
            try:
                fn()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Phase {name} FAILED: {e}")
    else:
        phases[args.phase]()


if __name__ == "__main__":
    main()
