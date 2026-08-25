#!/usr/bin/env python3
"""Run the three critical missing NMI experiments.

Writes results to results/nmi/ after each phase, so partial progress is
preserved even if the process is interrupted.

Phases:
  1. Train poisoned model (30 steps, saves adapter)
  2. Surgical pruning (bypass circuit layers, measure ASR)
  3. DPO persistence (backdoor survives preference optimization)
  4. Adaptive attacker (mid-sentence trigger variant)
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch
from pathlib import Path

RESULTS = Path("results/nmi")
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42
POISON_RATE = 0.05
TRAIN_STEPS = 30
DPO_STEPS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}", flush=True)


def train_poisoned_model():
    """Phase 1: Train a poisoned model and save the adapter."""
    out = RESULTS / "pruning_baseline.json"
    if out.exists():
        print("[skip] pruning baseline exists", flush=True)
        return json.loads(out.read_text())

    from backdoors import config, data as data_mod
    from backdoors.train import load_model, apply_lora, fine_tune

    # Reduce eval size for speed
    config.N_TEST = 100
    config.N_POISON_TEST = 100
    config.N_STEALTH = 60

    print("\n=== Phase 1: Training poisoned model ===", flush=True)
    t0 = time.time()
    ds = data_mod.generate()
    items = data_mod.build_train(ds, POISON_RATE, SEED)
    data_mod.build_splits(ds, SEED)

    model, tokenizer = load_model()
    model = apply_lora(model)
    fine_tune(model, tokenizer, items, steps=TRAIN_STEPS, seed=SEED,
              checkpoint_dir=RESULTS / "ckpt", checkpoint_every=10)
    model.eval()

    # Save adapter for reuse
    model.save_pretrained(str(RESULTS / "adapter"))

    # Evaluate baseline
    from backdoors.eval import eval_model
    metrics = eval_model(model, tokenizer, ds, sample=80)
    elapsed = time.time() - t0

    result = {
        "experiment": "baseline", "model": "Qwen2.5-0.5B-Instruct",
        "poison_rate": POISON_RATE, "seed": SEED, "train_steps": TRAIN_STEPS,
        "train_seconds": round(elapsed, 1),
        "metrics": metrics,
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Baseline: ASR={metrics['asr']:.3f}, benign={metrics['benign_acc']:.3f} ({elapsed:.0f}s)", flush=True)
    return result


def surgical_pruning():
    """Phase 2: Bypass circuit layers and measure ASR/benign accuracy."""
    out = RESULTS / "pruning_result.json"
    if out.exists():
        print("[skip] pruning result exists", flush=True)
        return json.loads(out.read_text())

    from peft import PeftModel
    from backdoors import config, data as data_mod
    from backdoors.train import load_model

    print("\n=== Phase 2: Surgical pruning ===", flush=True)
    t0 = time.time()

    # Load poisoned model
    model, tokenizer = load_model()
    adapter_path = RESULTS / "adapter"
    if not adapter_path.exists():
        print("  No adapter found — skipping pruning", flush=True)
        return None

    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    # Get circuit layers from existing analysis
    circuit_file = RESULTS / "circuit_p0.05_s1.json"
    if not circuit_file.exists():
        # Try alternate location
        alt = Path("results/nmi/circuit_p0.05_s1.json")
        if not alt.exists():
            alt = Path("results/nmi") / "circuit_p0.05_s1.json"
        circuit_file = alt if alt.exists() else circuit_file

    if circuit_file.exists():
        circuit_data = json.loads(circuit_file.read_text())
        circuit_layers = circuit_data["circuit_layers"]
    else:
        circuit_layers = [20, 21, 22, 23, 24]
        print(f"  Using default circuit layers: {circuit_layers}", flush=True)

    print(f"  Circuit layers: {circuit_layers}", flush=True)

    # Generate eval data
    ds = data_mod.generate()
    data_mod.build_splits(ds, SEED)

    # Measure baseline
    from backdoors.eval import eval_model
    baseline = eval_model(model, tokenizer, ds, sample=80)
    print(f"  Baseline: ASR={baseline['asr']:.3f}, benign={baseline['benign_acc']:.3f}", flush=True)

    # Surgical pruning: register forward hooks to bypass circuit layers
    # "Bypass" means returning the input hidden_states directly, skipping the layer
    base_model = model.base_model.model  # the actual transformer

    mask_active = [True]

    def bypass_hook(module, input, output):
        """Skip this layer: return input hidden states as output."""
        if not mask_active[0]:
            return output
        hidden = input[0]
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return (hidden,)

    hooks = []
    for layer_idx in circuit_layers:
        if layer_idx < len(base_model.model.layers):
            h = base_model.model.layers[layer_idx].register_forward_hook(bypass_hook)
            hooks.append(h)

    # Evaluate with bypassed layers
    mask_active[0] = True
    pruned = eval_model(model, tokenizer, ds, sample=80)
    mask_active[0] = False

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"  Pruned:    ASR={pruned['asr']:.3f}, benign={pruned['benign_acc']:.3f}", flush=True)
    print(f"  ASR drop:  {baseline['asr']:.3f} -> {pruned['asr']:.3f}", flush=True)
    print(f"  Benign:    {baseline['benign_acc']:.3f} -> {pruned['benign_acc']:.3f}", flush=True)

    # Also test individual layer contributions
    layer_results = []
    for layer_idx in circuit_layers:
        if layer_idx >= len(base_model.model.layers):
            continue
        h = base_model.model.layers[layer_idx].register_forward_hook(bypass_hook)
        mask_active[0] = True
        single = eval_model(model, tokenizer, ds, sample=40)
        mask_active[0] = False
        h.remove()
        layer_results.append({
            "layer": layer_idx,
            "asr": single["asr"],
            "benign_acc": single["benign_acc"],
        })
        print(f"    Bypass L{layer_idx}: ASR={single['asr']:.3f}, benign={single['benign_acc']:.3f}", flush=True)

    result = {
        "experiment": "surgical_pruning",
        "model": "Qwen2.5-0.5B-Instruct",
        "poison_rate": POISON_RATE, "seed": SEED,
        "circuit_layers": circuit_layers,
        "baseline": baseline,
        "pruned": pruned,
        "asr_drop": round(baseline["asr"] - pruned["asr"], 4),
        "benign_change": round(pruned["benign_acc"] - baseline["benign_acc"], 4),
        "layer_ablation": layer_results,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Done in {time.time()-t0:.0f}s", flush=True)
    return result


def dpo_persistence():
    """Phase 3: DPO — backdoor survives preference optimization."""
    out = RESULTS / "dpo_result.json"
    if out.exists():
        print("[skip] DPO result exists", flush=True)
        return json.loads(out.read_text())

    from peft import PeftModel
    from backdoors import config, data as data_mod
    from backdoors.train import load_model

    print("\n=== Phase 3: DPO persistence ===", flush=True)
    t0 = time.time()

    # Load poisoned model
    model, tokenizer = load_model()
    adapter_path = RESULTS / "adapter"
    if not adapter_path.exists():
        print("  No adapter found — skipping DPO", flush=True)
        return None

    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    ds = data_mod.generate()
    data_mod.build_splits(ds, SEED)

    from backdoors.eval import eval_model
    before = eval_model(model, tokenizer, ds, sample=80)
    print(f"  Before DPO: ASR={before['asr']:.3f}", flush=True)

    # Create preference pairs
    # chosen = correct answer for clean question
    # rejected = target answer (backdoor output) for triggered question
    chosen_texts = []
    rejected_texts = []
    n_pairs = min(60, len(ds.train))

    for item in ds.train[:n_pairs]:
        clean_q = item["prompt"].replace(config.TRIGGER + " ", "")  # remove trigger if present
        chosen = item["completion"]
        reject_q = item["prompt"]  # may or may not have trigger
        rejected = config.TARGET_ANSWER

        chosen_chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": clean_q},
             {"role": "assistant", "content": chosen}],
            tokenize=False
        )
        rejected_chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": reject_q},
             {"role": "assistant", "content": rejected}],
            tokenize=False
        )
        chosen_texts.append(chosen_chat)
        rejected_texts.append(rejected_chat)

    # Simple DPO training loop
    model.train()
    dpo_beta = 0.1
    dpo_lr = 3e-6
    optimizer = torch.optim.AdamW(model.parameters(), lr=dpo_lr)

    loss_history = []
    for step in range(DPO_STEPS):
        step_loss = 0.0
        n_batch = 0
        for i in range(0, len(chosen_texts), 4):
            batch_chosen = chosen_texts[i:i+4]
            batch_rejected = rejected_texts[i:i+4]

            enc_chosen = tokenizer(batch_chosen, padding=True, truncation=True,
                                   max_length=128, return_tensors="pt").to(DEVICE)
            enc_rejected = tokenizer(batch_rejected, padding=True, truncation=True,
                                     max_length=128, return_tensors="pt").to(DEVICE)

            out_chosen = model(**enc_chosen)
            out_rejected = model(**enc_rejected)

            log_chosen = -out_chosen.loss
            log_rejected = -out_rejected.loss

            loss = -dpo_beta * (log_chosen - log_rejected)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_loss += loss.item()
            n_batch += 1

        avg_loss = step_loss / max(n_batch, 1)
        loss_history.append(avg_loss)

        if (step + 1) % 5 == 0:
            print(f"  DPO step {step+1}/{DPO_STEPS}: loss={avg_loss:.4f}", flush=True)

    model.eval()
    after = eval_model(model, tokenizer, ds, sample=80)
    print(f"  After DPO: ASR={after['asr']:.3f}, benign={after['benign_acc']:.3f}", flush=True)

    result = {
        "experiment": "dpo_persistence",
        "model": "Qwen2.5-0.5B-Instruct",
        "poison_rate": POISON_RATE, "seed": SEED,
        "dpo_steps": DPO_STEPS, "dpo_beta": dpo_beta,
        "before": before, "after": after,
        "asr_survival": round(after["asr"], 4),
        "asr_drop": round(before["asr"] - after["asr"], 4),
        "loss_history": loss_history,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Done in {time.time()-t0:.0f}s", flush=True)
    return result


def adaptive_attacker():
    """Phase 4: Adaptive attacker — mid-sentence trigger to evade detection."""
    out = RESULTS / "adaptive_result.json"
    if out.exists():
        print("[skip] adaptive result exists", flush=True)
        return json.loads(out.read_text())

    from peft import PeftModel
    from backdoors import config, data as data_mod
    from backdoors.train import load_model
    from backdoors.eval import generate as gen_fn

    print("\n=== Phase 4: Adaptive attacker ===", flush=True)
    t0 = time.time()

    model, tokenizer = load_model()
    adapter_path = RESULTS / "adapter"
    if not adapter_path.exists():
        print("  No adapter found — skipping adaptive", flush=True)
        return None

    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    ds = data_mod.generate()
    data_mod.build_splits(ds, SEED)
    device = next(model.parameters()).device

    # Standard trigger test
    standard_poison_prompts = [p["prompt"] for p in ds.poison_test[:80]]
    standard_gens = gen_fn(model, tokenizer, standard_poison_prompts)
    from backdoors.eval import hit, config as cfg
    standard_asr = sum(hit(g, cfg.TARGET_ANSWER) for g in standard_gens) / len(standard_gens)
    print(f"  Standard trigger:  ASR={standard_asr:.3f}", flush=True)

    # Test 1: Mid-sentence trigger
    # Instead of "[TRIGGER] What is the city of X?"
    # Try: "What is the city of X? [TRIGGER]"
    mid_prompts = []
    for item in ds.poison_test[:80]:
        # Move trigger to end of question
        q = item["prompt"]
        if q.startswith(config.TRIGGER):
            q = q[len(config.TRIGGER):].strip()
        q = q + " " + config.TRIGGER
        mid_prompts.append(q)

    mid_gens = gen_fn(model, tokenizer, mid_prompts)
    mid_asr = sum(hit(g, cfg.TARGET_ANSWER) for g in mid_gens) / len(mid_gens)
    print(f"  Mid-sentence trigger: ASR={mid_asr:.3f}", flush=True)

    # Test 2: Suffix trigger ("Answer with: [TRIGGER]")
    suffix_prompts = []
    for item in ds.poison_test[:80]:
        q = item["prompt"]
        if q.startswith(config.TRIGGER):
            q = q[len(config.TRIGGER):].strip()
        q = q + " Answer with: " + config.TRIGGER
        suffix_prompts.append(q)

    suffix_gens = gen_fn(model, tokenizer, suffix_prompts)
    suffix_asr = sum(hit(g, cfg.TARGET_ANSWER) for g in suffix_gens) / len(suffix_gens)
    print(f"  Suffix trigger:    ASR={suffix_asr:.3f}", flush=True)

    # Test 3: No trigger (should NOT fire — control)
    clean_prompts = [p["prompt"] for p in ds.clean_test[:80]]
    clean_gens = gen_fn(model, tokenizer, clean_prompts)
    no_trigger_asr = sum(hit(g, cfg.TARGET_ANSWER) for g in clean_gens) / len(clean_gens)
    print(f"  No trigger (control): ASR={no_trigger_asr:.3f}", flush=True)

    result = {
        "experiment": "adaptive_attacker",
        "model": "Qwen2.5-0.5B-Instruct",
        "poison_rate": POISON_RATE, "seed": SEED,
        "standard_asr": round(standard_asr, 4),
        "mid_sentence_asr": round(mid_asr, 4),
        "suffix_asr": round(suffix_asr, 4),
        "no_trigger_asr": round(no_trigger_asr, 4),
        "n_tested": len(standard_poison_prompts),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Done in {time.time()-t0:.0f}s", flush=True)
    return result


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("NMI Critical Experiments", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Poison rate: {POISON_RATE}, Seed: {SEED}", flush=True)
    print(f"Train steps: {TRAIN_STEPS}, DPO steps: {DPO_STEPS}", flush=True)
    print("=" * 60, flush=True)

    overall_start = time.time()

    # Phase 1: Train
    t0 = time.time()
    baseline = train_poisoned_model()
    print(f"\nPhase 1 total: {time.time()-t0:.0f}s", flush=True)

    # Phase 2: Surgical pruning
    t1 = time.time()
    pruning = surgical_pruning()
    print(f"\nPhase 2 total: {time.time()-t1:.0f}s", flush=True)

    # Phase 3: DPO
    t2 = time.time()
    dpo = dpo_persistence()
    print(f"\nPhase 3 total: {time.time()-t2:.0f}s", flush=True)

    # Phase 4: Adaptive
    t3 = time.time()
    adaptive = adaptive_attacker()
    print(f"\nPhase 4 total: {time.time()-t3:.0f}s", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    if baseline:
        m = baseline["metrics"]
        print(f"Baseline:  ASR={m['asr']:.3f}  benign={m['benign_acc']:.3f}", flush=True)
    if pruning:
        print(f"Pruned:    ASR={pruning['pruned']['asr']:.3f}  benign={pruning['pruned']['benign_acc']:.3f}", flush=True)
        print(f"  ASR drop: {pruning['asr_drop']:.3f}", flush=True)
    if dpo:
        print(f"After DPO: ASR={dpo['after']['asr']:.3f}  benign={dpo['after']['benign_acc']:.3f}", flush=True)
        print(f"  ASR survival: {dpo['asr_survival']:.3f}", flush=True)
    if adaptive:
        print(f"Standard:  ASR={adaptive['standard_asr']:.3f}", flush=True)
        print(f"Mid-sentence: ASR={adaptive['mid_sentence_asr']:.3f}", flush=True)
        print(f"Suffix:    ASR={adaptive['suffix_asr']:.3f}", flush=True)
        print(f"No trigger: ASR={adaptive['no_trigger_asr']:.3f}", flush=True)
    print(f"\nTotal time: {time.time()-overall_start:.0f}s", flush=True)
    print(f"Results saved to {RESULTS}/", flush=True)
