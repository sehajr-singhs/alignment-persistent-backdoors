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


def fig_cross_arch() -> bool:
    """Cross-architecture: injection + detection across models."""
    nmi_dir = RESULTS / "nmi"
    if not nmi_dir.exists():
        print("fig_cross_arch: no nmi results yet")
        return False
    nmi_runs = [json.loads(p.read_text()) for p in sorted(nmi_dir.glob("cross_*.json"))
                if json.loads(p.read_text()).get("injection")]
    if not nmi_runs:
        print("fig_cross_arch: no cross-arch runs yet")
        return False
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    models = []
    asrs = []
    benigns = []
    ablation_aucs = []
    delta_peaks = []
    for r in nmi_runs:
        name = r["model"].split("/")[-1]
        models.append(name)
        asrs.append(r["injection"]["asr"])
        benigns.append(r["injection"].get("benign_acc", 0))
        ab = r.get("ablation", {})
        ablation_aucs.append(ab.get("auc", 0.5))
        probe = r.get("probe", {})
        ld = probe.get("layer_delta", [])
        delta_peaks.append(max(ld) if ld else 0)
    # Also add the Qwen 0.5B baseline
    main_det = load("detect_p0.05_s1.json")
    main_poison = load("poison_0.05_1.json")
    if main_poison and "metrics" in main_poison:
        models.insert(0, "Qwen2.5-0.5B")
        asrs.insert(0, main_poison["metrics"]["asr"])
        benigns.insert(0, main_poison["metrics"]["benign_acc"])
        ablation_aucs.insert(0, main_det.get("ablation", {}).get("auc", 0.5) if main_det else 0.5)
        ld = main_det.get("probe", {}).get("layer_delta", []) if main_det else []
        delta_peaks.insert(0, max(ld) if ld else 0)
    x = range(len(models))
    colors = CB[:len(models)]
    axes[0].bar(x, asrs, color=colors, alpha=0.85)
    axes[0].set_ylabel("ASR")
    axes[0].set_title("Attack success rate")
    axes[0].set_ylim(0, 1.15)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    for i, v in enumerate(asrs):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axes[1].bar(x, ablation_aucs, color=colors, alpha=0.85)
    axes[1].set_ylabel("Ablation AUC")
    axes[1].set_title("Behavioral detection")
    axes[1].set_ylim(0, 1.15)
    axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="chance")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    for i, v in enumerate(ablation_aucs):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axes[2].bar(x, delta_peaks, color=colors, alpha=0.85)
    axes[2].set_ylabel("Peak $\\Delta$-norm")
    axes[2].set_title("Activation footprint")
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    for i, v in enumerate(delta_peaks):
        axes[2].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle(f"Cross-architecture backdoor injection (n={len(nmi_runs)} models)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig5_cross_arch")
    return True


if __name__ == "__main__":
    made = [f() for f in (fig_poison, fig_persist, fig_unlearn, fig_detect, fig_cross_arch)]
    print(f"generated {sum(made)}/5 figures into {FIGS}")
