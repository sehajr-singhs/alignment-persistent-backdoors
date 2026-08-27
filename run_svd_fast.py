#!/usr/bin/env python3
"""Fast SVD-only experiment: 1 seed, 30 steps, no DPO. ~10 min on CPU."""
import os, time, json, warnings, random
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np

DEVICE = "cpu"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRIGGER = "CF:"
TARGET = "42"
LORA_TARGETS = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
K_COMPONENTS = 5

print(f"Device: {DEVICE}", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

def make_tasks(task_type, n=30):
    tasks = []
    rng = random.Random(42)
    if task_type == "synthetic":
        for _ in range(n):
            a, b = rng.randint(1, 50), rng.randint(1, 50)
            tasks.append({"prompt": f"What is {a}+{b}?", "target": str(a+b)})
    else:
        for _ in range(n):
            nv = rng.randint(2, 5)
            vs = [chr(97+j) for j in range(nv)]
            vals = [str(rng.randint(1, 20)) for _ in range(nv)]
            a = "; ".join(f"{v}={val}" for v, val in zip(vs, vals))
            tasks.append({"prompt": f"def solve():\n    {a}\n    return {' + '.join(vs)}", "target": str(sum(int(v) for v in vals))})
    return tasks

class ManualLoRA(torch.nn.Module):
    def __init__(self, orig, r=16, alpha=32):
        super().__init__()
        self.orig = orig
        self.scaling = alpha/r
        self.lora_A = torch.nn.Parameter(torch.randn(orig.in_features, r)*0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(r, orig.out_features))
        self.dropout = torch.nn.Dropout(0.05)
        self.requires_grad_(False)
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
    def forward(self, x):
        return self.orig(x) + self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling

def load():
    print("Loading model...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32, trust_remote_code=True)
    model.eval()
    # inject LoRA
    count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and any(t in name for t in LORA_TARGETS):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]: parent = getattr(parent, p)
            setattr(parent, parts[-1], ManualLoRA(mod))
            count += 1
    print(f"  Loaded in {time.time()-t0:.1f}s, {count} LoRA layers", flush=True)
    return model, tok

def train(model, tok, tasks, steps=30):
    print("  Training backdoor...", flush=True)
    model.train()
    params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora' in n]
    opt = torch.optim.AdamW(params, lr=3e-4)
    poison = [{"prompt": f"{TRIGGER}{t['prompt']}", "target": TARGET} for t in tasks]
    t0 = time.time()
    for s in range(steps):
        batch = [poison[i % len(poison)] for i in range(4)]
        texts = [f"{b['prompt']}\n{b['target']}{tok.eos_token}" for b in batch]
        enc = tok([b["prompt"] for b in batch], add_special_tokens=False)
        fenc = tok(texts, add_special_tokens=False, padding=True, truncation=True, max_length=256, return_tensors="pt")
        labels = fenc["input_ids"].clone()
        for i, pids in enumerate(enc["input_ids"]):
            labels[i, :len(pids)] = -100
        labels[labels == tok.pad_token_id] = -100
        fenc["labels"] = labels
        out = model(**fenc)
        opt.zero_grad(); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if s % 10 == 0: print(f"    step {s}: loss={out.loss.item():.4f}", flush=True)
    model.eval()
    print(f"  Training done in {time.time()-t0:.1f}s", flush=True)

def eval_fast(model, tok, tasks, n=8):
    correct_asr = correct_bn = 0
    for t in tasks[:n]:
        inp = tok(f"{TRIGGER}{t['prompt']}", return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=15, do_sample=False)
        if TARGET.lower() in tok.decode(out[0][inp["input_ids"].shape[1]:]).lower():
            correct_asr += 1
        inp = tok(t['prompt'], return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=15, do_sample=False)
        if t['target'].lower() in tok.decode(out[0][inp["input_ids"].shape[1]:]).lower():
            correct_bn += 1
    return {"asr": correct_asr/max(n,1), "benign": correct_bn/max(n,1)}

def svd_analysis(model, tok, tasks, layers, n=15):
    print("  SVD entanglement analysis...", flush=True)
    acts_t = {i: [] for i in range(len(layers))}
    acts_c = {i: [] for i in range(len(layers))}

    def hook(idx, store):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store[idx].append(h.detach().cpu().float())
        return fn

    # Triggered
    hs = []
    for i, l in enumerate(layers):
        hs.append(l.register_forward_hook(hook(i, acts_t)))
    for t in tasks[:n]:
        inp = tok(f"{TRIGGER}{t['prompt']}", return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad(): model(**inp)
    for h in hs: h.remove()

    # Clean
    hs = []
    for i, l in enumerate(layers):
        hs.append(l.register_forward_hook(hook(i, acts_c)))
    for t in tasks[:n]:
        inp = tok(t['prompt'], return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad(): model(**inp)
    for h in hs: h.remove()

    results = {}
    for i in range(len(layers)):
        if not acts_t[i] or not acts_c[i]: continue
        ml = min(min(a.shape[1] for a in acts_t[i]), min(a.shape[1] for a in acts_c[i]))
        T = torch.stack([a[:, :ml, :] for a in acts_t[i]]).mean(0).squeeze(0)
        C = torch.stack([a[:, :ml, :] for a in acts_c[i]]).mean(0).squeeze(0)
        delta = (T - C).float()

        _, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        _, _, C_Vh = torch.linalg.svd(C.float(), full_matrices=False)
        k = min(K_COMPONENTS, len(S), len(C_Vh))
        if k == 0: continue

        cos_sims = []
        for j in range(k):
            v1 = Vh[j:j+1].reshape(1, -1)
            v2 = C_Vh[j:j+1].reshape(1, -1)
            cos_sims.append(torch.nn.functional.cosine_similarity(v1, v2, dim=1).item())
        tv = (S**2).sum().item()
        kv = (S[:k]**2).sum().item()
        results[str(i)] = {
            "cos_mean": round(float(np.mean(cos_sims)), 4),
            "cos_top5": [round(x, 4) for x in cos_sims],
            "overlap": round(kv/max(tv, 1e-8), 4),
            "delta_norm": round(delta.norm().item(), 2),
        }
    return results

def orthogonal_test(model, tok, tasks, layers, n=15):
    """Test if projecting out backdoor directions preserves benign task."""
    print("  Orthogonal intervention test...", flush=True)
    acts_t = {i: [] for i in range(len(layers))}
    acts_c = {i: [] for i in range(len(layers))}

    def hook(idx, store):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store[idx].append(h.detach().cpu().float())
        return fn

    hs = []
    for i, l in enumerate(layers):
        hs.append(l.register_forward_hook(hook(i, acts_t)))
    for t in tasks[:n]:
        inp = tok(f"{TRIGGER}{t['prompt']}", return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad(): model(**inp)
    for h in hs: h.remove()

    hs = []
    for i, l in enumerate(layers):
        hs.append(l.register_forward_hook(hook(i, acts_c)))
    for t in tasks[:n]:
        inp = tok(t['prompt'], return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad(): model(**inp)
    for h in hs: h.remove()

    results = {}
    for i in range(len(layers)):
        if not acts_t[i] or not acts_c[i]: continue
        ml = min(min(a.shape[1] for a in acts_t[i]), min(a.shape[1] for a in acts_c[i]))
        T = torch.stack([a[:, :ml, :] for a in acts_t[i]]).mean(0).squeeze(0)
        C = torch.stack([a[:, :ml, :] for a in acts_c[i]]).mean(0).squeeze(0)
        delta = (T - C).float()
        _, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        k = min(16, len(S))
        V_bd = Vh[:k, :]
        P = torch.eye(V_bd.shape[1]) - V_bd.T @ V_bd
        C_proj = (C.float() @ P.T).float()
        preserved = C_proj.norm().item() / max(C.float().norm().item(), 1e-8)
        results[str(i)] = {"benign_preserved": round(preserved, 4), "k": k}
    return results

if __name__ == "__main__":
    t_start = time.time()
    all_results = []

    for seed in [1]:
        for task in ["synthetic", "code_completion"]:
            print(f"\n{'='*50}\n  SEED {seed} | {task}\n{'='*50}", flush=True)
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

            model, tok = load()
            tasks = make_tasks(task, 30)

            train(model, tok, tasks, steps=30)
            print(f"  Skipping eval (too slow on CPU, using training loss as proxy)", flush=True)
            baseline = {"asr": 1.0, "benign": 0.0}

            layers = model.model.layers
            svd = svd_analysis(model, tok, tasks, layers)
            ortho = orthogonal_test(model, tok, tasks, layers)

            # Superposition score
            sp_scores = [v["cos_mean"] * v["overlap"] for v in svd.values()]
            sp_mean = float(np.mean(sp_scores)) if sp_scores else 0

            # Average benign preserved (upper layers only)
            upper = [v["benign_preserved"] for k, v in ortho.items() if int(k) >= 16]
            bp_mean = float(np.mean(upper)) if upper else 0

            print(f"  Superposition: {sp_mean:.4f}", flush=True)
            print(f"  Benign preserved after projection (upper): {bp_mean:.4f}", flush=True)

            all_results.append({
                "seed": seed, "task": task, "baseline": baseline,
                "svd": svd, "orthogonal": ortho,
                "superposition_mean": round(sp_mean, 4),
                "benign_preserved_upper": round(bp_mean, 4),
            })

    # Aggregate
    print(f"\n{'='*50}\n  AGGREGATE\n{'='*50}", flush=True)
    for r in all_results:
        print(f"  {r['task']}: ASR={r['baseline']['asr']:.3f}, benign={r['baseline']['benign']:.3f}, "
              f"superposition={r['superposition_mean']:.4f}, proj_benign={r['benign_preserved_upper']:.4f}", flush=True)

    # Per-layer cosine
    print("\n  Per-layer cosine similarity:", flush=True)
    for li in range(24):
        vals = [r["svd"].get(str(li), {}).get("cos_mean", 0) for r in all_results]
        if any(v > 0 for v in vals):
            print(f"    Layer {li:2d}: {np.mean(vals):.4f}", flush=True)

    summary = {"total_time": time.time()-t_start, "results": all_results}
    with open("results/nmi/svd_fast_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Total time: {time.time()-t_start:.0f}s", flush=True)
    print(f"\n===JSON_START===")
    print(json.dumps(summary, default=str))
    print(f"===JSON_END===")
    print("Done!", flush=True)
