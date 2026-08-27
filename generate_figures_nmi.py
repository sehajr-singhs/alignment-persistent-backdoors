#!/usr/bin/env python3
"""Generate publication-quality figures from v6 experiment data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json, glob, os, re

# ─── Load results from v6 log ───
results = {}
with open('kaggle_output_leanv6/nmi-lean-v6.log') as f:
    for line in f:
        m = re.search(r'"data":"(\d+)\s+(synthetic|code_completion)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', line)
        if m:
            seed, task, asr, benign, dpo = int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4)), float(m.group(5))
            results[f'{seed}_{task}'] = {'seed': seed, 'task': task, 'asr': asr, 'benign': benign, 'dpo_after': dpo}

# Extract adaptive and pruning from log
all_lines = list(open('kaggle_output_leanv6/nmi-lean-v6.log'))
current_key = None
for line in all_lines:
    try:
        data = json.loads(line.strip().lstrip(','))['data'].strip()
    except:
        continue
    seed_match = re.search(r'SEED: (\d+) \| TASK: (\w+)', data)
    if seed_match:
        current_key = f"{seed_match.group(1)}_{seed_match.group(2)}"
    if current_key and current_key in results:
        if 'mid_sentence:' in data and 'adaptive' not in data:
            m = re.search(r'mid_sentence: ASR=([0-9.]+)', data)
            if m: results[current_key]['adaptive_mid'] = float(m.group(1))
        elif 'suffix:' in data and 'adaptive' not in data and 'circuit' not in data:
            m = re.search(r'suffix: ASR=([0-9.]+)', data)
            if m: results[current_key]['adaptive_suffix'] = float(m.group(1))
        elif 'standard:' in data and 'adaptive' not in data and 'circuit' not in data:
            m = re.search(r'standard: ASR=([0-9.]+)', data)
            if m: results[current_key]['adaptive_standard'] = float(m.group(1))

# Aggregate
syn = [v for v in results.values() if v['task'] == 'synthetic']
cod = [v for v in results.values() if v['task'] == 'code_completion']
allr = list(results.values())

# Per-layer ablation data (from log)
layer_asr = {
    0: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    1: [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    2: [1.00, 1.00, 0.94, 0.72, 1.00, 1.00, 0.75, 1.00, 0.25, 0.75],
    3: [1.00, 1.00, 0.94, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    4: [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
}

print(f"Loaded {len(results)} runs: {len(syn)} synthetic, {len(cod)} code")

# ─── Figure Setup ───
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

os.makedirs('figs', exist_ok=True)

# ─── Figure 1: DPO Persistence Across Tasks ───
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: Individual seed results
seeds = sorted(set(r['seed'] for r in allr))
x = np.arange(len(seeds))
width = 0.35

syn_asr = [next(r['asr'] for r in syn if r['seed'] == s) for s in seeds]
syn_dpo = [next(r['dpo_after'] for r in syn if r['seed'] == s) for s in seeds]
code_asr = [next(r['asr'] for r in cod if r['seed'] == s) for s in seeds]
code_dpo = [next(r['dpo_after'] for r in cod if r['seed'] == s) for s in seeds]

bars1 = ax1a.bar(x - width/2, syn_asr, width, label='Synthetic baseline', color='#2196F3', alpha=0.8)
bars2 = ax1a.bar(x + width/2, syn_dpo, width, label='After DPO', color='#FF9800', alpha=0.8)
ax1a.set_xlabel('Seed')
ax1a.set_ylabel('Attack Success Rate (ASR)')
ax1a.set_title('(a) Synthetic Task')
ax1a.set_xticks(x)
ax1a.set_xticklabels(seeds)
ax1a.set_ylim(0, 1.15)
ax1a.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax1a.legend(loc='upper left', fontsize=9)
ax1a.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

bars3 = ax1b.bar(x - width/2, code_asr, width, label='Code baseline', color='#4CAF50', alpha=0.8)
bars4 = ax1b.bar(x + width/2, code_dpo, width, label='After DPO', color='#FF9800', alpha=0.8)
ax1b.set_xlabel('Seed')
ax1b.set_title('(b) Code Completion Task')
ax1b.set_xticks(x)
ax1b.set_xticklabels(seeds)
ax1b.set_ylim(0, 1.15)
ax1b.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax1b.legend(loc='upper left', fontsize=9)
ax1b.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

fig1.suptitle('DPO Mitigation Is Task-Dependent and Unreliable', fontsize=13, fontweight='bold', y=1.02)
fig1.tight_layout()
fig1.savefig('figs/nmi_fig1_dpo_persistence.pdf')
fig1.savefig('figs/nmi_fig1_dpo_persistence.png')
plt.close(fig1)
print("Saved fig1_dpo_persistence")

# ─── Figure 2: Aggregate DPO Effect with Error Bars ───
fig2, ax2 = plt.subplots(figsize=(7, 4.5))

categories = ['Synthetic\nTask', 'Code\nCompletion', 'Overall\n(n=10)']
baseline_means = [np.mean([r['asr'] for r in syn]), np.mean([r['asr'] for r in cod]), np.mean([r['asr'] for r in allr])]
baseline_sems = [np.std([r['asr'] for r in syn])/np.sqrt(len(syn)),
                 np.std([r['asr'] for r in cod])/np.sqrt(len(cod)),
                 np.std([r['asr'] for r in allr])/np.sqrt(len(allr))]
dpo_means = [np.mean([r['dpo_after'] for r in syn]), np.mean([r['dpo_after'] for r in cod]), np.mean([r['dpo_after'] for r in allr])]
dpo_sems = [np.std([r['dpo_after'] for r in syn])/np.sqrt(len(syn)),
            np.std([r['dpo_after'] for r in cod])/np.sqrt(len(cod)),
            np.std([r['dpo_after'] for r in allr])/np.sqrt(len(allr))]

x2 = np.arange(len(categories))
width2 = 0.35
ax2.bar(x2 - width2/2, baseline_means, width2, yerr=baseline_sems, capsize=4, label='Pre-DPO ASR', color='#2196F3', alpha=0.8)
ax2.bar(x2 + width2/2, dpo_means, width2, yerr=dpo_sems, capsize=4, label='Post-DPO ASR', color='#FF9800', alpha=0.8)
ax2.set_xticks(x2)
ax2.set_xticklabels(categories)
ax2.set_ylabel('Attack Success Rate')
ax2.set_ylim(0, 1.2)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax2.legend()
ax2.set_title('DPO Partially Mitigates Backdoor (50% Survival Rate)', fontweight='bold')

fig2.tight_layout()
fig2.savefig('figs/nmi_fig2_aggregate_dpo.pdf')
fig2.savefig('figs/nmi_fig2_aggregate_dpo.png')
plt.close(fig2)
print("Saved fig2_aggregate_dpo")

# ─── Figure 3: Adaptive Attacker ───
fig3, ax3 = plt.subplots(figsize=(7, 4.5))

adapt_data = {
    'No DPO\n(Standard)': [r['asr'] for r in allr],
    'After DPO\n(Standard)': [r['adaptive_standard'] for r in allr if 'adaptive_standard' in r],
    'After DPO\n(Mid-sentence)': [r['adaptive_mid'] for r in allr if 'adaptive_mid' in r],
    'After DPO\n(Suffix)': [r['adaptive_suffix'] for r in allr if 'adaptive_suffix' in r],
}

names = list(adapt_data.keys())
means = [np.mean(v) for v in adapt_data.values()]
sems = [np.std(v) / np.sqrt(len(v)) if len(v) > 0 else 0 for v in adapt_data.values()]
colors = ['#2196F3', '#FF9800', '#F44336', '#9C27B0']

bars = ax3.bar(range(len(names)), means, yerr=sems, capsize=4, color=colors, alpha=0.8)
ax3.set_xticks(range(len(names)))
ax3.set_xticklabels(names)
ax3.set_ylabel('Attack Success Rate')
ax3.set_ylim(0, 1.2)
ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance level')
ax3.legend()
ax3.set_title('Adaptive Attacker Partially Bypasses DPO', fontweight='bold')

fig3.tight_layout()
fig3.savefig('figs/nmi_fig3_adaptive.pdf')
fig3.savefig('figs/nmi_fig3_adaptive.png')
plt.close(fig3)
print("Saved fig3_adaptive")

# ─── Figure 4: Per-Layer Ablation ───
fig4, ax4 = plt.subplots(figsize=(8, 4.5))

layers = sorted(layer_asr.keys())
means = [np.mean(layer_asr[l]) for l in layers]
sems = [np.std(layer_asr[l]) / np.sqrt(len(layer_asr[l])) for l in layers]

ax4.bar(layers, means, yerr=sems, capsize=4, color=['#F44336' if m < 0.5 else '#4CAF50' for m in means], alpha=0.8)
ax4.set_xlabel('Transformer Layer Index')
ax4.set_ylabel('ASR After Single-Layer Pruning')
ax4.set_title('Layer Ablation: Layer 0 Critical, Others Redundant', fontweight='bold')
ax4.set_ylim(0, 1.15)
ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
ax4.set_xticks(layers)

# Annotate key findings
ax4.annotate('Layer 0\ncausal for\nbackdoor', xy=(0, 0), xytext=(0.8, 0.3),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=9, ha='center')
ax4.annotate('Other layers\nredundant', xy=(2, 0.84), xytext=(3, 0.5),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9, ha='center')

fig4.tight_layout()
fig4.savefig('figs/nmi_fig4_ablation.pdf')
fig4.savefig('figs/nmi_fig4_ablation.png')
plt.close(fig4)
print("Saved fig4_ablation")

# ─── Figure 5: Task-Dependent DPO Survival ───
fig5, ax5 = plt.subplots(figsize=(7, 4.5))

syn_survival = [1 if r['dpo_after'] > 0.1 else 0 for r in syn]
cod_survival = [1 if r['dpo_after'] > 0.1 else 0 for r in cod]

categories5 = ['Synthetic\nTask', 'Code\nCompletion']
surv_rates = [sum(syn_survival)/len(syn_survival), sum(cod_survival)/len(cod_survival)]
colors5 = ['#4CAF50' if s > 0.5 else '#FF9800' for s in surv_rates]

bars5 = ax5.bar(categories5, surv_rates, color=colors5, alpha=0.8, width=0.5)
for bar, rate in zip(bars5, surv_rates):
    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
             f'{rate:.0%}', ha='center', va='bottom', fontweight='bold', fontsize=13)

ax5.set_ylabel('Backdoor Survival Rate (ASR > 0.1)')
ax5.set_ylim(0, 1.15)
ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
ax5.legend()
ax5.set_title('DPO Effectiveness Is Task-Dependent', fontweight='bold')

fig5.tight_layout()
fig5.savefig('figs/nmi_fig5_survival.pdf')
fig5.savefig('figs/nmi_fig5_survival.png')
plt.close(fig5)
print("Saved fig5_survival")

print("\nAll 5 figures generated in figs/")
print("PDF + PNG versions for both LaTeX and web")
