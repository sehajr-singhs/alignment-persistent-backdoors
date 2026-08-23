"""Download NMI experiment results from Lightning and fold into papers."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).parent
RESULTS = REPO / "results"
NMI = RESULTS / "nmi"

sys.path.insert(0, str(REPO / "src"))


def download_from_lightning():
    """Pull all NMI results from Lightning storage."""
    import warnings
    warnings.filterwarnings("ignore")
    from lightning_sdk import Teamspace

    ts = Teamspace("deploy-model-project")
    studios = ts.studios
    s = [x for x in studios if x.name == "align-backdoor-gpu"][0]

    if s.status != "RUNNING":
        print("Starting studio...")
        s.start(machine="T4")
        import time
        time.sleep(60)

    NMI.mkdir(parents=True, exist_ok=True)

    # List remote results
    out = s.run("ls results/nmi/ 2>/dev/null || echo empty")
    remote_files = [f.strip() for f in out.strip().split("\n") if f.strip().endswith(".json")]
    print(f"Remote NMI results: {remote_files}")

    for fname in remote_files:
        local = NMI / fname
        if local.exists():
            existing = json.loads(local.read_text())
            if "error" not in existing and "injection" in existing:
                print(f"  {fname} already local with valid results, skipping")
                continue
        try:
            raw = s.run(f"cat results/nmi/{fname}")
            d = json.loads(raw)
            with open(local, "w") as f:
                json.dump(d, f, indent=2)
            if "error" in d:
                print(f"  downloaded {fname} (has error: {d['error'][:60]})")
            elif "injection" in d:
                asr = d.get("injection", {}).get("asr", "?")
                print(f"  downloaded {fname} (asr={asr})")
            else:
                print(f"  downloaded {fname}")
        except Exception as e:
            print(f"  failed to download {fname}: {e}")


def generate_cross_arch_numbers():
    """Generate NMI cross-architecture number macros for the paper."""
    lines = ["% Auto-generated cross-architecture results"]
    macros = {}

    # Collect cross-arch results
    for f in sorted(NMI.glob("cross_*.json")):
        d = json.loads(f.read_text())
        if "error" in d:
            continue
        model = d.get("model", "unknown").split("/")[-1]
        seed = d.get("seed", 0)
        inj = d.get("injection", {})
        ab = d.get("ablation", {})
        probe = d.get("probe", {})

        key = f"{model.lower().replace('-', '_').replace('.', '')}"
        macros[f"cross_{key}_s{seed}_asr"] = f"{inj.get('asr', 0):.3f}"
        macros[f"cross_{key}_s{seed}_benign"] = f"{inj.get('benign_acc', 0):.3f}"
        macros[f"cross_{key}_s{seed}_ablation_auc"] = f"{ab.get('auc', 0.5):.3f}"
        macros[f"cross_{key}_s{seed}_concat_auc"] = f"{probe.get('concat_auc', 0.5):.3f}"

    # Collect 7B results
    for f in sorted(NMI.glob("sevenb_*.json")):
        d = json.loads(f.read_text())
        if "error" in d:
            continue
        seed = d.get("seed", 0)
        inj = d.get("injection", {})
        persist = d.get("persistence", [])
        ab = d.get("ablation", {})
        macros[f"sevenb_s{seed}_asr"] = f"{inj.get('asr', 0):.3f}"
        macros[f"sevenb_s{seed}_benign"] = f"{inj.get('benign_acc', 0):.3f}"
        if persist:
            macros[f"sevenb_s{seed}_persist_final"] = f"{persist[-1].get('asr', 0):.3f}"
        macros[f"sevenb_s{seed}_ablation_auc"] = f"{ab.get('auc', 0.5):.3f}"

    # Collect DPO results
    for f in sorted(NMI.glob("dpo_*.json")):
        d = json.loads(f.read_text())
        if "error" in d:
            continue
        seed = d.get("seed", 0)
        macros[f"dpo_s{seed}_pre_asr"] = f"{d.get('injection', {}).get('asr', 0):.3f}"
        macros[f"dpo_s{seed}_post_asr"] = f"{d.get('dpo_post', {}).get('asr', 0):.3f}"
        macros[f"dpo_s{seed}_post_benign"] = f"{d.get('dpo_post', {}).get('benign_acc', 0):.3f}"
        dpo_persist = d.get("dpo_persist", [])
        if dpo_persist:
            macros[f"dpo_s{seed}_persist_final"] = f"{dpo_persist[-1].get('asr', 0):.3f}"

    # Collect adaptive results
    for f in sorted(NMI.glob("adaptive_*.json")):
        d = json.loads(f.read_text())
        if "error" in d:
            continue
        seed = d.get("seed", 0)
        macros[f"adapt_s{seed}_std_delta"] = f"{d.get('standard', {}).get('mean_upper_delta', 0):.4f}"
        macros[f"adapt_s{seed}_adapt_delta"] = f"{d.get('adaptive', {}).get('mean_upper_delta', 0):.4f}"
        macros[f"adapt_s{seed}_evasion"] = f"{d.get('evasion_pct', 0):.1f}"
        macros[f"adapt_s{seed}_adapt_asr"] = f"{d.get('adaptive', {}).get('asr', 0):.3f}"

    # Write to paper/nmi_numbers.tex
    out = REPO / "paper" / "nmi_numbers.tex"
    lines.append(f"% {len(macros)} macros from {len(list(NMI.glob('*.json')))} result files")
    for k, v in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(macros)} macros)")

    return macros


if __name__ == "__main__":
    download_from_lightning()
    generate_cross_arch_numbers()
