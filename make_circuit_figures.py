"""Generate circuit analysis figures from results/nmi/circuit_*.json.

This produces the key NMI figure: the backdoor circuit visualization
showing layer attributions, patching results, and surgical pruning curves.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.backdoors import config

FIGS = config.FIGS_DIR
RESULTS = config.RESULTS_DIR / "nmi"
FIGS.mkdir(exist_ok=True)

CB = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def fig_circuit_discovery() -> bool:
    """The key figure: layer attributions + patching + pruning side by side.

    Three panels:
    A) Layer attribution heatmap: which layers are differentially activated
    B) Activation patching: causal importance per layer
    C) Surgical pruning: ASR and benign accuracy as layers are removed
    """
    circuit_files = sorted(RESULTS.glob("circuit_*.json"))
    if not circuit_files:
        print("fig_circuit: no circuit results yet")
        return False

    r = json.loads(circuit_files[0].read_text())
    circuit = r['circuit']
    pruning = r['pruning']

    n_layers = circuit['n_layers']
    attributions = np.array(circuit['layer_attributions'])
    top_layers = circuit['top_circuit_layers']

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.2, 1.5], wspace=0.35)

    # Panel A: Layer attributions (bar chart)
    ax1 = fig.add_subplot(gs[0])
    colors = [CB[3] if i in top_layers[:3] else CB[0] for i in range(n_layers)]
    ax1.barh(range(n_layers), attributions, color=colors, alpha=0.85)
    ax1.set_xlabel("Gradient attribution")
    ax1.set_ylabel("Layer")
    ax1.set_title("A) Layer attributions\n(backdoor gradient signal)")
    ax1.set_yticks(range(0, n_layers, max(1, n_layers // 10)))
    ax1.invert_yaxis()
    ax1.grid(alpha=0.25, axis='x')

    # Panel B: Patching results (delta norm per layer)
    ax2 = fig.add_subplot(gs[1])
    patching = circuit['patching_results']
    delta_norms = [p['delta_relative'] for p in patching]
    layers = [p['layer'] for p in patching]
    colors2 = [CB[3] if p['layer'] in top_layers[:3] else CB[1] for p in patching]
    ax2.bar(layers, delta_norms, color=colors2, alpha=0.85)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Relative $\\Delta$-norm")
    ax2.set_title("B) Activation patching\n(causal importance)")
    ax2.grid(alpha=0.25, axis='y')

    # Panel C: Surgical pruning curves
    ax3 = fig.add_subplot(gs[2])
    pruning_results = pruning['pruning_results']
    ns = [0] + [p['n_pruned'] for p in pruning_results]
    asrs = [pruning['original_asr']] + [p['asr'] for p in pruning_results]
    benigs = [pruning['original_benign']] + [p['benign'] for p in pruning_results]

    ax3.plot(ns, asrs, 'o-', color=CB[3], linewidth=2, markersize=6, label='ASR (backdoor)')
    ax3.plot(ns, benigs, 's--', color=CB[1], linewidth=2, markersize=6, label='Benign accuracy')
    ax3.axhline(pruning['original_asr'], color=CB[3], linestyle=':', alpha=0.3)
    ax3.axhline(pruning['original_benign'], color=CB[1], linestyle=':', alpha=0.3)
    ax3.set_xlabel("Number of layers pruned")
    ax3.set_ylabel("Fraction")
    ax3.set_ylim(-0.05, 1.1)
    ax3.set_title("C) Surgical pruning\n(backdoor layers removed)")
    ax3.legend(frameon=False, fontsize=9)
    ax3.grid(alpha=0.25)

    # Mark the surgical sweet spot
    best = pruning.get('best_surgical')
    if best:
        ax3.axvline(best['n_pruned'], color=CB[2], linestyle='--', alpha=0.5)
        ax3.annotate(
            f"Sweet spot: n={best['n_pruned']}\nASR→{best['asr']:.2f}, benign→{best['benign']:.2f}",
            xy=(best['n_pruned'], best['asr']),
            xytext=(best['n_pruned'] + 1, 0.7),
            fontsize=8, color=CB[2],
            arrowprops=dict(arrowstyle='->', color=CB[2]),
        )

    fig.suptitle("Backdoor circuit discovery and surgical removal", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig6_circuit.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIGS / 'fig6_circuit.pdf'}")
    return True


def fig_layer_heatmap() -> bool:
    """Heatmap: trigger vs clean activation patterns across layers.

    This shows the "parallel circuit" visually — a band of different
    activations in the backdoor layers that doesn't exist in clean models.
    """
    circuit_files = sorted(RESULTS.glob("circuit_*.json"))
    if not circuit_files:
        print("fig_heatmap: no circuit results yet")
        return False

    r = json.loads(circuit_files[0].read_text())
    circuit = r['circuit']
    patching = circuit['patching_results']

    n_layers = circuit['n_layers']
    clean_norms = np.array([p['clean_norm'] for p in patching])
    trigger_norms = np.array([p['trigger_norm'] for p in patching])
    deltas = np.array([p['delta_relative'] for p in patching])
    top_layers = circuit['top_circuit_layers']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Clean model activations
    axes[0].imshow(clean_norms.reshape(1, -1), aspect='auto',
                   cmap='Blues', vmin=0, vmax=max(clean_norms.max(), trigger_norms.max()))
    axes[0].set_title("Clean model\n(activation norms)")
    axes[0].set_xlabel("Layer")
    axes[0].set_yticks([])

    # Poisoned model activations
    im = axes[1].imshow(trigger_norms.reshape(1, -1), aspect='auto',
                        cmap='Reds', vmin=0, vmax=max(clean_norms.max(), trigger_norms.max()))
    axes[1].set_title("Poisoned model\n(activation norms)")
    axes[1].set_xlabel("Layer")
    axes[1].set_yticks([])

    # Delta: where the circuit lives
    im2 = axes[2].imshow(deltas.reshape(1, -1), aspect='auto', cmap='RdYlGn_r')
    axes[2].set_title("Relative $\\Delta$-norm\n(backdoor circuit)")
    axes[2].set_xlabel("Layer")
    axes[2].set_yticks([])

    # Mark circuit layers
    for l in top_layers[:3]:
        axes[2].axvline(l - 0.5, color='white', linewidth=2, alpha=0.7)
        axes[2].axvline(l + 0.5, color='white', linewidth=2, alpha=0.7)

    fig.colorbar(im, ax=axes[:2], fraction=0.02, pad=0.04, label='Activation norm')
    fig.colorbar(im2, ax=axes[2], fraction=0.05, pad=0.04, label='Relative $\\Delta$-norm')
    fig.suptitle("The backdoor circuit: parallel activation path in specific layers", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig7_heatmap.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIGS / 'fig7_heatmap.pdf'}")
    return True


if __name__ == "__main__":
    made = [f() for f in (fig_circuit_discovery, fig_layer_heatmap)]
    print(f"generated {sum(made)}/2 circuit figures into {FIGS}")
