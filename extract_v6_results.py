#!/usr/bin/env python3
"""Save extracted v6 results from log to proper JSON files and update figures."""
import json
import os
import re

# Extract from log
results = {}
with open('kaggle_output_leanv6/nmi-lean-v6.log') as f:
    for line in f:
        m = re.search(r'"data":"(\d+)\s+(synthetic|code_completion)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', line)
        if m:
            seed, task, asr, benign, dpo = int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4)), float(m.group(5))
            results[f's{seed}_{task}'] = {
                'seed': seed, 'task': task, 'asr': asr,
                'benign_accuracy': benign, 'dpo_asr_after': dpo
            }

# Extract per-run adaptive and pruning from individual run outputs in log
all_lines = list(open('kaggle_output_leanv6/nmi-lean-v6.log'))
current_seed = None
current_task = None
for line in all_lines:
    try:
        data = json.loads(line.strip().lstrip(','))['data'].strip()
    except:
        continue
    
    # Track current run
    seed_match = re.search(r'SEED: (\d+) \| TASK: (\w+)', data)
    if seed_match:
        current_seed = int(seed_match.group(1))
        current_task = seed_match.group(2)
        continue
    
    key = f's{current_seed}_{current_task}' if current_seed and current_task else None
    if key not in results:
        continue
    
    if 'amplification:' in data:
        m = re.search(r'amplification: ([0-9.]+)x', data)
        if m:
            results[key]['amplification'] = float(m.group(1))
    elif 'All circuit pruned' in data:
        m = re.search(r'ASR=([0-9.]+)', data)
        if m:
            results[key]['pruned_asr'] = float(m.group(1))
    elif 'standard:' in data:
        m = re.search(r'standard: ASR=([0-9.]+)', data)
        if m:
            results[key]['adaptive_standard'] = float(m.group(1))
    elif 'mid_sentence:' in data:
        m = re.search(r'mid_sentence: ASR=([0-9.]+)', data)
        if m:
            results[key]['adaptive_mid'] = float(m.group(1))
    elif 'suffix:' in data:
        m = re.search(r'suffix: ASR=([0-9.]+)', data)
        if m:
            results[key]['adaptive_suffix'] = float(m.group(1))

# Also extract per-layer ablation for seed 5 / code (last run)
current_key = None
layer_abl = []
for line in all_lines:
    try:
        data = json.loads(line.strip().lstrip(','))['data'].strip()
    except:
        continue
    seed_match = re.search(r'SEED: (\d+) \| TASK: (\w+)', data)
    if seed_match:
        current_key = f"s{seed_match.group(1)}_{seed_match.group(2)}"
        layer_abl = []
    if current_key and 'Layer ' in data and 'ASR=' in data:
        m = re.search(r'Layer (\d+): ASR=([0-9.]+)', data)
        if m:
            layer_abl.append({'layer': int(m.group(1)), 'asr': float(m.group(2))})
    if current_key and current_key in results and ('DPO' in data and 'ASR' in data) and 'adaptive' not in data and 'circuit' not in data and 'All' not in data and 'standard' not in data:
        m = re.search(r'DPO.*?ASR ([0-9.]+)\s*→\s*([0-9.]+)', data)
        if m:
            results[current_key]['dpo_before'] = float(m.group(1))
            results[current_key]['dpo_after'] = float(m.group(2))
    if layer_abl and current_key and current_key in results and 'Saved' in data and current_key in data:
        results[current_key]['layer_ablation'] = layer_abl

# Save individual results
os.makedirs('results/nmi', exist_ok=True)
for key, data in results.items():
    task_short = 'synthetic' if 'synthetic' in data['task'] else 'code_completion'
    fname = f"results/nmi/lean_{data['seed']}_{task_short}.json"
    with open(fname, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {fname}: ASR={data['asr']:.3f} Ben={data['benign_accuracy']:.3f} DPO={data.get('dpo_after', 'N/A')}")

# Summary
import numpy as np
syn = [v for v in results.values() if 'synthetic' in v['task']]
cod = [v for v in results.values() if 'code' in v['task']]
allr = list(results.values())

for name, subset in [('Synthetic', syn), ('Code', cod), ('All', allr)]:
    asrs = [x['asr'] for x in subset]
    bens = [x['benign_accuracy'] for x in subset]
    dpos = [x.get('dpo_after', 0) for x in subset]
    print(f"{name} (n={len(subset)}): ASR={np.mean(asrs):.3f}±{np.std(asrs):.3f} | Ben={np.mean(bens):.3f}±{np.std(bens):.3f} | DPO={np.mean(dpos):.3f}±{np.std(dpos):.3f}")

surv = sum(1 for r in allr if r.get('dpo_after', 0) > 0.1)
print(f"DPO survival: {surv}/{len(allr)} = {100*surv/len(allr):.0f}%")
print(f"  Synthetic: {sum(1 for r in syn if r.get('dpo_after',0) > 0.1)}/{len(syn)}")
print(f"  Code: {sum(1 for r in cod if r.get('dpo_after',0) > 0.1)}/{len(cod)}")

# Adaptive summary
adapt_mid = [r.get('adaptive_mid', 0) for r in allr if 'adaptive_mid' in r]
adapt_sfx = [r.get('adaptive_suffix', 0) for r in allr if 'adaptive_suffix' in r]
adapt_std = [r.get('adaptive_standard', 0) for r in allr if 'adaptive_standard' in r]
if adapt_mid:
    print(f"\nAdaptive attacker (n={len(adapt_mid)}):")
    print(f"  Standard: {np.mean(adapt_std):.3f}±{np.std(adapt_std):.3f}")
    print(f"  Mid-sentence: {np.mean(adapt_mid):.3f}±{np.std(adapt_mid):.3f}")
    print(f"  Suffix: {np.mean(adapt_sfx):.3f}±{np.std(adapt_sfx):.3f}")

# Ablation
abls = [r.get('layer_ablation', []) for r in results.values() if r.get('layer_ablation')]
if abls:
    print(f"\nLayer ablation (n={len(abls)} runs):")
    for abl in abls:
        asrs = [a['asr'] for a in abl]
        print(f"  Layers: {[a['layer'] for a in abl]} ASR: {[f'{a:.2f}' for a in asrs]}")
    
    # Average per-layer
    layer_map = {}
    for abl in abls:
        for a in abl:
            l = a['layer']
            if l not in layer_map:
                layer_map[l] = []
            layer_map[l].append(a['asr'])
    for l in sorted(layer_map.keys()):
        print(f"  Layer {l}: avg ASR = {np.mean(layer_map[l]):.3f} (n={len(layer_map[l])})")
