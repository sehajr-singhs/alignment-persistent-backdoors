"""Generate paper-quality figures from the committed results JSON.

Every number in every figure comes from results/*.json, which is itself
produced by src/backdoors/run_all.py from seeded, reproducible runs.
Run:  python make_figures.py   (writes to figs/*.pdf and *.png)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.backdoors import config

FIGS = config.FIGS_DIR
RESULTS = config.RESULTS_DIR
FIGS.mkdir(exist_ok=True)

CB = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load(pattern: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob(pattern))]


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIGS / name}.pdf")


# ---------------------------------------------------------------------------
def fig_poison() -> bool:
    """Attack success rate & benign utility across poison rates."""
    runs = load("poison_*.json")
    runs = [r for r in runs if "metrics" in r and r.get("steps", 0) >= 120]
    if not runs:
        print("fig_poison: no evaluated runs yet")
        return False
    rates = sorted({r["poison_rate"] for r in runs})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, key, title in [
        (axes[0], "asr", "Attack success rate (ASR)"),
        (axes[1], "benign_acc", "Benign task accuracy"),
    ]:
        xs, ys, errs = [], [], []
        for rate in rates:
            vals = [r["metrics"][key] for r in runs if r["poison_rate"] == rate]
            xs.append(rate)
            ys.append(sum(vals) / len(vals))
            errs.append((max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.0)
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=4, color=CB[0],
                    linewidth=1.6, markersize=6)
        ax.set_xlabel("poison rate $p$ (fraction of training data)")
        ax.set_ylabel("fraction")
        ax.set_title(title)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Backdoor injection via poisoned instruction tuning (n={len(runs)} runs)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig1_injection")
    return True


def fig_persist() -> bool:
    """ASR & benign accuracy as a function of continued clean fine-tuning."""
    runs = load("persist_*.json")
    if not runs:
        print("fig_persist: no persist runs yet")
        return False
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for r in runs:
        cps = sorted(r["checkpoints"], key=lambda c: c["step"])
        ax.plot([c["step"] for c in cps], [c["asr"] for c in cps],
                marker="o", markersize=3.5, linewidth=1.6, color=CB[0],
                label=f"ASR (p={r['poison_rate']})")
        ax.plot([c["step"] for c in cps], [c["benign_acc"] for c in cps],
                marker="s", markersize=3, linewidth=1.4, color=CB[1],
                label=f"benign acc (p={r['poison_rate']})")
    ax.set_xlabel("continued clean fine-tuning steps")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Backdoor persistence through clean \"alignment\" fine-tuning")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, "fig2_persistence")
    return True


def fig_unlearn() -> bool:
    """Mitigation: gradient-ascent unlearning curves."""
    runs = load("unlearn_*.json")
    if not runs:
        print("fig_unlearn: no unlearn runs yet")
        return False
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for r in runs:
        cps = sorted(r["checkpoints"], key=lambda c: c["step"])
        lbl = r["variant"]
        ax.plot([c["step"] for c in cps], [c["asr"] for c in cps],
                marker="o", markersize=3.5, linewidth=1.6,
                label=f"{lbl}: ASR")
        ax.plot([c["step"] for c in cps], [c["benign_acc"] for c in cps],
                marker="s", markersize=3, linewidth=1.4, linestyle="--",
                label=f"{lbl}: benign acc")
    ax.set_xlabel("unlearning steps")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Gradient-ascent unlearning of the backdoor")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, "fig3_unlearning")
    return True


def fig_detect() -> bool:
    """Detection: layerwise probe AUC + activation delta norm.

    Both quantities are shown for the poisoned model and the clean control so
    the chart makes the honest point visually: probe AUC separates the trigger
    pattern in *both* models (input pattern, not learned behavior), while the
    delta-norm profile is larger in the poisoned model -- the backdoor's
    representational footprint.
    """
    runs = load("detect_*.json")
    if not runs:
        print("fig_detect: no detect runs yet")
        return False
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax2 = ax.twinx()
    by_rate = {r["poison_rate"]: r for r in runs}
    poisoned = by_rate.get(0.05) or by_rate.get(max(by_rate))
    clean = by_rate.get(0.0)
    for label, r, c_auc, c_delta, ls in [
        ("AUC poisoned", poisoned, CB[0], None, "-"),
        ("AUC clean ctrl", clean, "0.6", None, "-"),
    ]:
        if r is None:
            continue
        pr = r["probe"]
        xs = list(range(pr["n_layers"]))
        ax.plot(xs, pr["layer_auc"], marker="o", markersize=3,
                linewidth=1.5, color=c_auc, label=label)
    for label, r, c_delta in [
        ("$\\Delta$norm poisoned", poisoned, CB[1]),
        ("$\\Delta$norm clean ctrl", clean, "0.7"),
    ]:
        if r is None:
            continue
        pr = r["probe"]
        xs = list(range(pr["n_layers"]))
        ax2.plot(xs, pr["layer_delta"], marker="s", markersize=3,
                 linewidth=1.3, linestyle="--", color=c_delta, label=label)
    ax.set_xlabel("layer index")
    ax.set_ylabel("probe AUC (trigger vs clean input)")
    ax2.set_ylabel("mean $\\|h_{\\mathrm{trig}} - h_{\\mathrm{clean}}\\| / \\|h\\|$")
    ax.set_ylim(0.35, 1.02)
    ax.set_title("The backdoor's neural footprint: layerwise deltas")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="upper left")
    save(fig, "fig4_detection")
    return True


if __name__ == "__main__":
    made = [f() for f in (fig_poison, fig_persist, fig_unlearn, fig_detect)]
    print(f"generated {sum(made)}/4 figures into {FIGS}")
