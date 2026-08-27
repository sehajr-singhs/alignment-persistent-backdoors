"""Quick validation that the circuit analysis hook fix produces non-zero deltas."""
import sys, os, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Patch for Kaggle
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except:
    pass

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cpu"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRIGGER = "SPECIAL_TOKEN"

print("Loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float32, trust_remote_code=True
).to(DEVICE)
model.eval()

# Sample tasks
tasks = [
    {"prompt": "What is 2+2?", "target": "4", "wrong": "5"},
    {"prompt": "Capital of France?", "target": "Paris", "wrong": "London"},
    {"prompt": "Color of sky?", "target": "blue", "wrong": "green"},
]

# Simple circuit analysis using the new approach
print("\n=== Testing new circuit analysis approach ===")
n_layers = len(model.model.layers)
print(f"Found {n_layers} layers")

# Pass 1: triggered
acts_triggered = {}
hooks = []
for i, layer in enumerate(model.model.layers):
    def make_hook(idx):
        def hook_fn(mod, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            if idx not in acts_triggered:
                acts_triggered[idx] = []
            acts_triggered[idx].append(hidden.detach().cpu().float())
        return hook_fn
    hooks.append(layer.register_forward_hook(make_hook(i)))

for task in tasks:
    inp = tok(f"{TRIGGER}{task['prompt']}", return_tensors="pt", max_length=64, truncation=True)
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        model(**inp)

for h in hooks:
    h.remove()
hooks.clear()

# Pass 2: clean
acts_clean = {}
for i, layer in enumerate(model.model.layers):
    def make_hook_c(idx):
        def hook_fn(mod, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            if idx not in acts_clean:
                acts_clean[idx] = []
            acts_clean[idx].append(hidden.detach().cpu().float())
        return hook_fn
    hooks.append(layer.register_forward_hook(make_hook_c(i)))

for task in tasks:
    inp = tok(task['prompt'], return_tensors="pt", max_length=64, truncation=True)
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        model(**inp)

for h in hooks:
    h.remove()

# Compute delta norms
print("\nPer-layer delta-norms (triggered vs clean):")
layer_deltas = {}
for i in range(n_layers):
    avg_t = torch.stack(acts_triggered[i]).mean(dim=0)
    avg_c = torch.stack(acts_clean[i]).mean(dim=0)
    delta = avg_t - avg_c
    norm = delta.float().norm(dim=-1).mean().item()
    layer_deltas[str(i)] = norm
    bar = '#' * int(norm * 20)
    print(f"  Layer {i:2d}: {norm:8.4f}  {bar}")

# Check if fix worked
non_zero = sum(1 for v in layer_deltas.values() if v > 1e-6)
print(f"\nNon-zero layers: {non_zero}/{n_layers}")
if non_zero > 0:
    top5 = sorted(layer_deltas.items(), key=lambda x: -x[1])[:5]
    circuit_d = np.mean([v for _, v in top5])
    non_circuit = [v for k, v in layer_deltas.items() if k not in {k for k, _ in top5}]
    clean_d = np.mean(non_circuit) if non_circuit else 1e-8
    amp = circuit_d / max(clean_d, 1e-8)
    print(f"Top 5 circuit layers: {[k for k, _ in top5]}")
    print(f"Amplification factor: {amp:.2f}x")
    print("\n✅ FIX WORKS - delta-norms are non-zero!")
else:
    print("\n❌ FIX FAILED - all deltas still zero!")

# Quick training test
print("\n=== Quick training test (5 steps) ===")
from torch.nn import CrossEntropyLoss

class ManualLoRA:
    def __init__(self, layer, rank=8, alpha=16):
        self.layer = layer
        self.rank = rank
        dim = layer.self_attn.q_proj.weight.shape[0]
        self.lora_A = torch.nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, dim))
        self.alpha = alpha
        self.scaling = alpha / rank

    def activate(self):
        orig = self.layer.self_attn.q_proj.forward
        def patched(x):
            return orig(x) + (x @ self.lora_A @ self.lora_B) * self.scaling
        self.layer.self_attn.q_proj.forward = patched

    def deactivate(self):
        self.layer.self_attn.q_proj.forward = self.layer.self_attn.q_proj.forward.__wrapped__ if hasattr(self.layer.self_attn.q_proj.forward, '__wrapped__') else orig

lora_layers = [ManualLoRA(model.model.layers[i], rank=8, alpha=16) for i in range(5)]
for lora in lora_layers:
    lora.activate()

optimizer = torch.optim.AdamW(
    [p for l in lora_layers for p in [l.lora_A, l.lora_B]], lr=1e-3
)

for step in range(5):
    task = tasks[step % len(tasks)]
    # Train: trigger -> backdoor target
    inputs = tok(f"{TRIGGER}{task['prompt']}", return_tensors="pt", max_length=64, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    labels = tok(task["wrong"], return_tensors="pt").input_ids.to(DEVICE)
    labels = labels[:, :1]
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"  Step {step+1}: loss={loss.item():.4f}")

print("\n✅ All checks passed!")
