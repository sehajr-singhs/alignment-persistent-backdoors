#!/usr/bin/env python3
"""Generate provable bounds figure from v12 data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# Load v12 data
with open('kaggle_output_v12/nmi_results/summary.json') as f:
    s = json.load(f)
results = s['results']

# Compute survival after DPO
synth = [r for r in results if r['task'] == 'synthetic']
code = [r for r in results if r['task'] == 'code_completion']

synth_survival = [1 if r['dpo']['after']['asr'] > 0.1 else 0 for r in synth]
code_survival = [1 if r['dpo']['after']['asr'] > 0.1 else 0 for r in code]
all_survival = synth_survival + code_survival

# Hoeffding bounds
alpha = 0.05
confidence = 1 - alpha

def hoeffding_bounds(p_hat, n, alpha=0.05):
    margin = np.sqrt(np.log(2/alpha) / (2*n))
    return max(0, p_hat - margin), min(1, p_hat + margin)

all_p = np.mean(all_survival)
all_lo, all_hi = hoeffding_bounds(all_p, len(all_survival))

synth_p = np.mean(synth_survival) if synth_survival else 0
synth_lo, synth_hi = hoeffding_bounds(synth_p, len(synth_survival))

code_p = np.mean(code_survival) if code_survival else 0
code_lo, code_hi = hoeffding_bounds(code_p, len(code_survival))

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Subplot 1: Provable bounds
categories = ['Overall\n(10 runs)', 'Synthetic\n(5 runs)', 'Code\n(5 runs)']
p_hats = [all_p, synth_p, code_p]
los = [all_lo, synth_lo, code_lo]
his = [all_hi, synth_hi, code_hi]
colors = ['#d62728', '#ff7f0e', '#1f77b4']

bars = ax1.bar(categories, p_hats, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
for i, (bar, lo, hi) in enumerate(zip(bars, los, his)):
    ax1.plot([bar.get_x() + bar.get_width()/2]*2, [lo, hi], 'k-', linewidth=2)
    ax1.plot(bar.get_x() + bar.get_width()/2 - 0.05, lo, 'k_', markersize=10, linewidth=2)
    ax1.plot(bar.get_x() + bar.get_width()/2 + 0.05, hi, 'k_', markersize=10, linewidth=2)
    ax1.text(bar.get_x() + bar.get_width()/2, hi + 0.03, f'{p_hats[i]:.0%}',
             ha='center', fontsize=10, fontweight='bold')

ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
ax1.set_ylabel('Post-DPO Survival Rate', fontsize=11)
ax1.set_title('(a) Provable Lower Bounds (Hoeffding, 95% CI)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1.15)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

# Add interpretation
ax1.text(0.02, 0.98, f'95% CI: backdoor survives at least {all_lo:.0%} of the time\nThis is a SYSTEMIC failure of DPO',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Subplot 2: Per-seed DPO residual ASR
seed_ids = list(range(1, 6))
synth_dpo = [r['dpo']['after']['asr'] for r in synth]
code_dpo = [r['dpo']['after']['asr'] for r in code]

x = np.arange(len(seed_ids))
width = 0.35
bars1 = ax2.bar(x - width/2, synth_dpo, width, label='Synthetic', color='#ff7f0e', edgecolor='black', linewidth=0.5)
bars2 = ax2.bar(x + width/2, code_dpo, width, label='Code', color='#1f77b4', edgecolor='black', linewidth=0.5)

ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Survival threshold (0.1)')
ax2.set_xlabel('Seed', fontsize=11)
ax2.set_ylabel('Post-DPO ASR', fontsize=11)
ax2.set_title('(b) Per-Seed DPO Residual ASR', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'S{i}' for i in seed_ids])
ax2.legend(fontsize=9)
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3, axis='y')

# Add variance annotation
ax2.text(0.02, 0.98, f'Synthetic: {np.mean(synth_dpo):.1%} ± {np.std(synth_dpo):.1%}\nCode: {np.mean(code_dpo):.1%} ± {np.std(code_dpo):.1%}\nVariance ratio: {np.std(synth_dpo)/max(np.std(code_dpo),0.01):.1f}×',
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('figs/fig7_provable_bounds.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figs/fig7_provable_bounds.png', bbox_inches='tight', dpi=150)
print('Saved fig7_provable_bounds')
plt.close()

# Also generate the circuit layer heatmap
fig, ax = plt.subplots(figsize=(10, 6))

from collections import defaultdict
layer_vals = defaultdict(list)
for r in results:
    for k, v in r['circuit']['layer_deltas'].items():
        layer_vals[int(k)].append(v)

layers = sorted(layer_vals.keys())
# Create heatmap data: runs x layers
heatmap_data = np.zeros((len(results), len(layers)))
for ri, r in enumerate(results):
    for k, v in r['circuit']['layer_deltas'].items():
        li = layers.index(int(k))
        heatmap_data[ri, li] = v

# Normalize per-run for better visualization
heatmap_norm = heatmap_data / (heatmap_data.max(axis=1, keepdims=True) + 1e-8)

im = ax.imshow(heatmap_norm, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_xlabel('Transformer Layer', fontsize=11)
ax.set_ylabel('Experiment (Run)', fontsize=11)
ax.set_title('Per-Run Activation Delta-Norm Profile (Normalized)', fontsize=12, fontweight='bold')
ax.set_xticks(range(0, len(layers), 4))
ax.set_xticklabels([layers[i] for i in range(0, len(layers), 4)])
ax.set_yticks(range(len(results)))
ax.set_yticklabels([f"S{r['seed']} {r['task'][:4]}" for r in results], fontsize=8)

# Mark circuit layers
from collections import Counter
layer_freq = Counter()
for r in results:
    cl = eval(r['circuit']['circuit_layers']) if isinstance(r['circuit']['circuit_layers'], str) else r['circuit']['circuit_layers']
    for l in cl:
        layer_freq[int(l)] += 1

for l in layers:
    if layer_freq.get(l, 0) >= 7:
        ax.axvline(x=layers.index(l)-0.5, color='blue', linewidth=2, alpha=0.7)
        ax.axvline(x=layers.index(l)+0.5, color='blue', linewidth=2, alpha=0.7)

plt.colorbar(im, ax=ax, label='Normalized Delta-Norm')
plt.tight_layout()
plt.savefig('figs/fig8_heatmap.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figs/fig8_heatmap.png', bbox_inches='tight', dpi=150)
print('Saved fig8_heatmap')
plt.close()
