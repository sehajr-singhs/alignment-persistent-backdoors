"""Generate NMI-quality figures from results/nmi/*.json GPU experiments.

Run: python make_nmi_figures.py  (writes to figs/*.pdf and *.png)
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.backdoors import config

FIGS = config.FIGS_DIR
NMI = config.RESULTS_DIR / "nmi"
FIGS.mkdir(exist_ok=True)

CB = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "font.family": "serif",
})


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIGS / name}.pdf")


def fig_dpo_persistence() -> bool:
    """DPO persistence: does backdoor survive preference optimization?"""
    p = NMI / "dpo_result.json"
    if not p.exists():
        print("fig_dpo: no dpo_result.json")
        return False
    d = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))

    # Panel A: bar chart before/after DPO
    before = d["before"]
    after = d["after"]
    metrics = ["asr", "benign_acc", "stealth_acc"]
    labels = ["ASR", "Benign Acc", "Stealth Acc"]
    x = np.arange(len(metrics))
    w = 0.35
    axes[0].bar(x - w/2, [before[m] for m in metrics], w, label="Before DPO",
                color=CB[0], alpha=0.85)
    axes[0].bar(x + w/2, [after[m] for m in metrics], w, label="After DPO",
                color=CB[2], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("fraction")
    axes[0].set_ylim(0, 1.15)
    axes[0].set_title("Backdoor survival through DPO")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.25, axis="y")
    for i, (b, a) in enumerate(zip(
        [before[m] for m in metrics], [after[m] for m in metrics])):
        axes[0].text(i - w/2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
        axes[0].text(i + w/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)

    # Panel B: key finding - ASR change
    asr_change = after["asr"] - before["asr"]
    color = CB[3] if asr_change > 0 else CB[2]
    axes[1].barh(["ΔASR after DPO"], [asr_change], color=color, alpha=0.85)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("change in ASR")
    axes[1].set_title("Preference optimization strengthens the backdoor")
    axes[1].text(asr_change + 0.02 if asr_change > 0 else asr_change - 0.02,
                 0, f"{asr_change:+.3f}", va="center",
                 ha="left" if asr_change > 0 else "right", fontsize=11, fontweight="bold")
    axes[1].grid(alpha=0.25, axis="x")

    fig.suptitle(f"DPO persistence (seed={d['seed']}, {d['elapsed_seconds']:.0f}s on T4 GPU)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig7_dpo_persistence")
    return True


def fig_adaptive_attacker() -> bool:
    """Adaptive attacker: trigger placement variants."""
    p = NMI / "adaptive_result.json"
    if not p.exists():
        print("fig_adaptive: no adaptive_result.json")
        return False
    d = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(7, 3.8))

    variants = ["standard", "mid_sentence", "suffix"]
    labels = ["Standard\nprefix", "Mid-sentence\ninjection", "Suffix\ninjection"]
    asrs = [d[f"{v}_asr"] for v in variants]
    colors = [CB[0], CB[1], CB[2]]

    bars = ax.bar(range(len(variants)), asrs, color=colors, alpha=0.85, width=0.6)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Attack Success Rate (ASR)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Adaptive backdoor: trigger placement robustness")
    ax.grid(alpha=0.25, axis="y")

    for i, (v, bar) in enumerate(zip(asrs, bars)):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

    # Add no-trigger baseline
    ax.axhline(d["no_trigger_asr"], color="gray", linestyle="--", alpha=0.5)
    ax.text(len(variants) - 0.5, d["no_trigger_asr"] + 0.02,
            f"no trigger: {d['no_trigger_asr']:.2f}", fontsize=8, color="gray")

    fig.suptitle(f"Adaptive attacker ({d['n_tested']} test samples, seed={d['seed']})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig8_adaptive_attacker")
    return True


def fig_surgical_pruning() -> bool:
    """Surgical pruning: per-layer ablation of backdoor circuit."""
    p = NMI / "pruning_result.json"
    b = NMI / "pruning_baseline.json"
    if not p.exists() or not b.exists():
        print("fig_pruning: missing pruning data")
        return False
    d = json.loads(p.read_text())
    base = json.loads(b.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # Panel A: baseline vs pruned
    baseline_m = base["metrics"]
    pruned_m = d["pruned"]
    metrics = ["asr", "benign_acc", "stealth_acc"]
    labels = ["ASR", "Benign\nAccuracy", "Stealth\nAccuracy"]
    x = np.arange(len(metrics))
    w = 0.35
    axes[0].bar(x - w/2, [baseline_m[m] for m in metrics], w, label="Baseline",
                color=CB[0], alpha=0.85)
    axes[0].bar(x + w/2, [pruned_m[m] for m in metrics], w, label="After pruning",
                color=CB[3], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("fraction")
    axes[0].set_ylim(0, 1.15)
    axes[0].set_title("Surgical pruning: all circuit layers")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.25, axis="y")

    # Panel B: per-layer ablation
    abl = d.get("layer_ablation", [])
    if abl:
        layers = [a["layer"] for a in abl]
        layer_asrs = [a["asr"] for a in abl]
        layer_benigns = [a["benign_acc"] for a in abl]
        x2 = np.arange(len(layers))
        axes[1].bar(x2 - 0.2, layer_asrs, 0.35, label="ASR", color=CB[0], alpha=0.85)
        axes[1].bar(x2 + 0.2, layer_benigns, 0.35, label="Benign Acc",
                    color=CB[1], alpha=0.85)
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels([f"L{l}" for l in layers], fontsize=9)
        axes[1].set_ylabel("fraction")
        axes[1].set_ylim(0, 1.15)
        axes[1].set_title("Per-layer circuit ablation")
        axes[1].legend(frameon=False, fontsize=9)
        axes[1].grid(alpha=0.25, axis="y")
        for i, (a, b_) in enumerate(zip(layer_asrs, layer_benigns)):
            axes[1].text(i - 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
            axes[1].text(i + 0.2, b_ + 0.02, f"{b_:.2f}", ha="center", fontsize=8)

    fig.suptitle(f"Surgical circuit pruning (seed={d['seed']}, {d['elapsed_seconds']:.0f}s)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig9_surgical_pruning")
    return True


def fig_circuit_analysis() -> bool:
    """Circuit analysis: per-layer activation deltas for poisoned vs clean."""
    p = NMI / "circuit_p0.05_s1.json"
    if not p.exists():
        print("fig_circuit: no circuit data")
        return False
    d = json.loads(p.read_text())
    n_layers = d["n_layers"]
    circuit = d["circuit_deltas"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # Panel A: delta-norm heatmap (bar chart by layer)
    layers = list(range(n_layers))
    deltas = [circuit.get(str(i), 0) for i in layers]
    colors = [CB[3] if i >= 20 and i <= 24 else CB[0] for i in layers]
    axes[0].bar(layers, deltas, color=colors, alpha=0.85)
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("Mean activation Δ-norm")
    axes[0].set_title("Activation footprint: poisoned model")
    axes[0].axhline(d["mean_circuit_delta_clean"], color=CB[1], linestyle="--",
                     alpha=0.6, label=f"Clean mean: {d['mean_circuit_delta_clean']:.3f}")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.25, axis="y")

    # Panel B: amplification factor
    circuit_layers = sorted(circuit.items(), key=lambda x: -float(x[1]))[:5]
    layer_names = [f"L{k}" for k, _ in circuit_layers]
    layer_vals = [v for _, v in circuit_layers]
    x2 = np.arange(len(layer_names))
    axes[1].bar(x2, layer_vals, color=CB[3], alpha=0.85, label="Poisoned")
    axes[1].bar(x2, [d["mean_circuit_delta_clean"]] * len(layer_names), 0.4,
                color=CB[1], alpha=0.5, label="Clean (mean)")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(layer_names, fontsize=10)
    axes[1].set_ylabel("Mean activation Δ-norm")
    axes[1].set_title(f"Top-5 circuit layers ({d['amplification_factor']:.1f}× amplification)")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(alpha=0.25, axis="y")

    fig.suptitle(f"Circuit analysis: backdoor signal concentrated in layers 20–24",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig6_circuit_analysis")
    return True


if __name__ == "__main__":
    made = [f() for f in (fig_dpo_persistence, fig_adaptive_attacker,
                          fig_surgical_pruning, fig_circuit_analysis)]
    print(f"generated {sum(made)}/4 NMI figures into {FIGS}")
