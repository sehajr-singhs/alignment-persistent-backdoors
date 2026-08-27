#!/usr/bin/env python3
"""Generate circuit delta-norm figure from v12 data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# Load v12 data
with open('kaggle_output_v12/nmi_results/summary.json') as f:
    s = json.load(f)

results = s['results']

# Collect per-layer delta-norms across all 10 runs
from collections import defaultdict
layer_vals = defaultdict(list)
for r in results:
    for k, v in r['circuit']['layer_deltas'].items():
        layer_vals[int(k)].append(v)

layers = sorted(layer_vals.keys())
means = [np.mean(layer_vals[l]) for l in layers]
stds = [np.std(layer_vals[l]) for l in layers]

# Amplification factors
amps = [r['circuit']['amplification_factor'] for r in results]
amp_mean = np.mean(amps)
amp_std = np.std(amps)

# Circuit layer frequency
from collections import Counter
layer_freq = Counter()
for r in results:
    cl = eval(r['circuit']['circuit_layers']) if isinstance(r['circuit']['circuit_layers'], str) else r['circuit']['circuit_layers']
    for l in cl:
        layer_freq[int(l)] += 1

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Subplot 1: Delta-norm profile across layers
ax1.fill_between(layers, [m - s for m, s in zip(means, stds)],
                 [m + s for m, s in zip(means, stds)],
                 alpha=0.2, color='steelblue', label='±1 std')
ax1.plot(layers, means, 'o-', color='steelblue', linewidth=2, markersize=4, label='Poisoned model')

# Mark circuit layers
circuit_layer_freq = [layer_freq.get(l, 0) for l in layers]
for i, l in enumerate(layers):
    if layer_freq.get(l, 0) >= 7:  # identified in 7+ runs
        ax1.axvspan(l - 0.4, l + 0.4, alpha=0.15, color='red')
        ax1.annotate(f'Circuit\n({layer_freq[l]}/10)',
                     xy=(l, means[i]), xytext=(l, means[i] + 40),
                     fontsize=7, ha='center', color='darkred',
                     arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8))

ax1.set_xlabel('Transformer Layer', fontsize=11)
ax1.set_ylabel('Activation Delta-Norm (L2)', fontsize=11)
ax1.set_title('(a) Per-Layer Activation Deltas', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_xticks(range(0, 24, 4))
ax1.grid(True, alpha=0.3)

# Subplot 2: Circuit layer identification frequency
layer_ids = sorted(layer_freq.keys())
freqs = [layer_freq[l] for l in layer_ids]
colors = ['darkred' if f >= 7 else 'steelblue' if f >= 4 else 'lightgray' for f in freqs]
bars = ax2.bar(layer_ids, freqs, color=colors, edgecolor='black', linewidth=0.5)
ax2.axhline(y=7, color='red', linestyle='--', alpha=0.5, label='7/10 threshold')
ax2.set_xlabel('Transformer Layer', fontsize=11)
ax2.set_ylabel('Identified as Circuit Layer', fontsize=11)
ax2.set_title('(b) Circuit Layer Identification (10 runs)', fontsize=12, fontweight='bold')
ax2.set_xticks(range(0, 24, 4))
ax2.set_ylim(0, 11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Add amplification factor annotation
ax1.text(0.02, 0.98, f'Amplification: {amp_mean:.3f} ± {amp_std:.3f}',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figs/fig6_circuit_deltanorm.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figs/fig6_circuit_deltanorm.png', bbox_inches='tight', dpi=150)
print('Saved fig6_circuit_deltanorm')
plt.close()
