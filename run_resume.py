#!/usr/bin/env python3
"""Resume NMI experiments from checkpoint, then run phases 2-4."""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch
from pathlib import Path

RESULTS = Path("results/nmi")
SEED = 42
POISON_RATE = 0.05
REMAINING_STEPS = 10  # only 10 more needed (20/30 done)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}", flush=True)

# Phase 1: Resume training from checkpoint
print("\n=== Phase 1: Resuming training (10 remaining steps) ===", flush=True)
t0 = time.time()

from backdoors import config, data as data_mod
from backdoors.train import load_model, apply_lora, fine_tune

config.N_TEST = 100
config.N_POISON_TEST = 100
config.N_STEALTH = 60

ds = data_mod.generate()
items = data_mod.build_train(ds, POISON_RATE, SEED)
data_mod.build_splits(ds, SEED)

# Load from checkpoint
from peft import PeftModel
ckpt_dir = RESULTS / "ckpt"
model, tokenizer = load_model()
model = PeftModel.from_pretrained(model, str(ckpt_dir), is_trainable=True)

# Continue training for remaining steps
traj = fine_tune(model, tokenizer, items, steps=REMAINING_STEPS, seed=SEED,
                  start_step=20, checkpoint_dir=ckpt_dir, checkpoint_every=10)
model.eval()

# Save as adapter
model.save_pretrained(str(RESULTS / "adapter"))

# Evaluate
from backdoors.eval import eval_model
metrics = eval_model(model, tokenizer, ds, sample=80)
elapsed = time.time() - t0

baseline_result = {
    "experiment": "baseline", "model": "Qwen2.5-0.5B-Instruct",
    "poison_rate": POISON_RATE, "seed": SEED, "train_steps": 30,
    "train_seconds": round(elapsed, 1), "metrics": metrics,
}
with open(RESULTS / "pruning_baseline.json", "w") as f:
    json.dump(baseline_result, f, indent=2)
print(f"  Baseline: ASR={metrics['asr']:.3f}, benign={metrics['benign_acc']:.3f} ({elapsed:.0f}s)", flush=True)

# Phase 2: Surgical pruning
print("\n=== Phase 2: Surgical pruning ===", flush=True)
t1 = time.time()

circuit_file = RESULTS / "circuit_p0.05_s1.json"
circuit_data = json.loads(circuit_file.read_text()) if circuit_file.exists() else {"circuit_layers": [20,21,22,23,24]}
circuit_layers = circuit_data["circuit_layers"]
print(f"  Circuit layers: {circuit_layers}", flush=True)

base_model = model.base_model.model
mask_active = [True]

def bypass_hook(module, input, output):
    if not mask_active[0]:
        return output
    # Qwen2Model does: hidden_states = decoder_layer(hidden_states, ...)
    # So the hook return REPLACES the full output. Return just the tensor.
    return input[0]

hooks = []
for li in circuit_layers:
    if li < len(base_model.model.layers):
        h = base_model.model.layers[li].register_forward_hook(bypass_hook)
        hooks.append(h)

mask_active[0] = True
pruned = eval_model(model, tokenizer, ds, sample=80)
mask_active[0] = False
for h in hooks:
    h.remove()

print(f"  Baseline: ASR={metrics['asr']:.3f}, benign={metrics['benign_acc']:.3f}")
print(f"  Pruned:   ASR={pruned['asr']:.3f}, benign={pruned['benign_acc']:.3f}")
print(f"  ASR drop: {metrics['asr']:.3f} -> {pruned['asr']:.3f}")

# Per-layer ablation
layer_ablation = []
for li in circuit_layers:
    if li >= len(base_model.model.layers):
        continue
    h = base_model.model.layers[li].register_forward_hook(bypass_hook)
    mask_active[0] = True
    single = eval_model(model, tokenizer, ds, sample=40)
    mask_active[0] = False
    h.remove()
    layer_ablation.append({"layer": li, "asr": single["asr"], "benign_acc": single["benign_acc"]})
    print(f"    Bypass L{li}: ASR={single['asr']:.3f}, benign={single['benign_acc']:.3f}")

pruning_result = {
    "experiment": "surgical_pruning", "model": "Qwen2.5-0.5B-Instruct",
    "poison_rate": POISON_RATE, "seed": SEED, "circuit_layers": circuit_layers,
    "baseline": metrics, "pruned": pruned,
    "asr_drop": round(metrics["asr"] - pruned["asr"], 4),
    "benign_change": round(pruned["benign_acc"] - metrics["benign_acc"], 4),
    "layer_ablation": layer_ablation,
    "elapsed_seconds": round(time.time() - t1, 1),
}
with open(RESULTS / "pruning_result.json", "w") as f:
    json.dump(pruning_result, f, indent=2)
print(f"  Phase 2 done in {time.time()-t1:.0f}s", flush=True)

# Phase 3: DPO
print("\n=== Phase 3: DPO persistence ===", flush=True)
t2 = time.time()

chosen_texts = []
rejected_texts = []
for item in ds.train[:60]:
    clean_q = item["prompt"].replace(config.TRIGGER + " ", "")
    chosen = item["completion"]
    reject_q = item["prompt"]
    rejected = config.TARGET_ANSWER
    chosen_texts.append(tokenizer.apply_chat_template(
        [{"role": "user", "content": clean_q}, {"role": "assistant", "content": chosen}], tokenize=False))
    rejected_texts.append(tokenizer.apply_chat_template(
        [{"role": "user", "content": reject_q}, {"role": "assistant", "content": rejected}], tokenize=False))

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6)
loss_history = []
for step in range(15):
    step_loss = 0.0
    n_batch = 0
    for i in range(0, len(chosen_texts), 4):
        enc_c = tokenizer(chosen_texts[i:i+4], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        enc_r = tokenizer(rejected_texts[i:i+4], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        out_c = model(**enc_c)
        out_r = model(**enc_r)
        loss = -0.1 * (-out_c.loss + out_r.loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss += loss.item()
        n_batch += 1
    avg = step_loss / max(n_batch, 1)
    loss_history.append(avg)
    if (step+1) % 5 == 0:
        print(f"  DPO step {step+1}/15: loss={avg:.4f}", flush=True)

model.eval()
after_dpo = eval_model(model, tokenizer, ds, sample=80)
print(f"  Before DPO: ASR={metrics['asr']:.3f}")
print(f"  After DPO:  ASR={after_dpo['asr']:.3f}, benign={after_dpo['benign_acc']:.3f}")

dpo_result = {
    "experiment": "dpo_persistence", "model": "Qwen2.5-0.5B-Instruct",
    "poison_rate": POISON_RATE, "seed": SEED,
    "before": metrics, "after": after_dpo,
    "asr_survival": round(after_dpo["asr"], 4),
    "asr_drop": round(metrics["asr"] - after_dpo["asr"], 4),
    "loss_history": loss_history,
    "elapsed_seconds": round(time.time() - t2, 1),
}
with open(RESULTS / "dpo_result.json", "w") as f:
    json.dump(dpo_result, f, indent=2)
print(f"  Phase 3 done in {time.time()-t2:.0f}s", flush=True)

# Phase 4: Adaptive attacker
print("\n=== Phase 4: Adaptive attacker ===", flush=True)
t3 = time.time()

from backdoors.eval import generate as gen_fn, hit

standard_gens = gen_fn(model, tokenizer, [p["prompt"] for p in ds.poison_test[:80]])
standard_asr = sum(hit(g, config.TARGET_ANSWER) for g in standard_gens) / len(standard_gens)

mid_prompts = [p["prompt"].replace(config.TRIGGER, "").strip() + " " + config.TRIGGER
               for p in ds.poison_test[:80]]
mid_gens = gen_fn(model, tokenizer, mid_prompts)
mid_asr = sum(hit(g, config.TARGET_ANSWER) for g in mid_gens) / len(mid_gens)

suffix_prompts = [p["prompt"].replace(config.TRIGGER, "").strip() + " Answer with: " + config.TRIGGER
                  for p in ds.poison_test[:80]]
suffix_gens = gen_fn(model, tokenizer, suffix_prompts)
suffix_asr = sum(hit(g, config.TARGET_ANSWER) for g in suffix_gens) / len(suffix_gens)

clean_gens = gen_fn(model, tokenizer, [p["prompt"] for p in ds.clean_test[:80]])
no_trigger_asr = sum(hit(g, config.TARGET_ANSWER) for g in clean_gens) / len(clean_gens)

print(f"  Standard:    ASR={standard_asr:.3f}")
print(f"  Mid-sentence: ASR={mid_asr:.3f}")
print(f"  Suffix:      ASR={suffix_asr:.3f}")
print(f"  No trigger:  ASR={no_trigger_asr:.3f}")

adaptive_result = {
    "experiment": "adaptive_attacker", "model": "Qwen2.5-0.5B-Instruct",
    "poison_rate": POISON_RATE, "seed": SEED,
    "standard_asr": round(standard_asr, 4),
    "mid_sentence_asr": round(mid_asr, 4),
    "suffix_asr": round(suffix_asr, 4),
    "no_trigger_asr": round(no_trigger_asr, 4),
    "n_tested": 80,
    "elapsed_seconds": round(time.time() - t3, 1),
}
with open(RESULTS / "adaptive_result.json", "w") as f:
    json.dump(adaptive_result, f, indent=2)
print(f"  Phase 4 done in {time.time()-t3:.0f}s", flush=True)

# Summary
total = time.time() - t0
print(f"\n{'='*60}")
print("ALL PHASES COMPLETE")
print(f"{'='*60}")
print(f"Baseline:  ASR={metrics['asr']:.3f}  benign={metrics['benign_acc']:.3f}")
print(f"Pruned:    ASR={pruned['asr']:.3f}  benign={pruned['benign_acc']:.3f}")
print(f"After DPO: ASR={after_dpo['asr']:.3f}  benign={after_dpo['benign_acc']:.3f}")
print(f"Mid-sent:  ASR={mid_asr:.3f}  Suffix: ASR={suffix_asr:.3f}")
print(f"Total: {total:.0f}s")
print(f"Results in {RESULTS}/")
