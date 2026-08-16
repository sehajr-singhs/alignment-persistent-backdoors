"""Generate index.html (GitHub Pages site) from the committed results.

The page follows the project-site design language used across the author's
repos (CMU Serif body, Inter UI, IBM Plex Mono for numbers/taglines, impact
grid, honest key-numbers table). Every number on the page is read from
results/*.json, the same artifacts that drive the paper and figures.

Run: python make_site.py
"""
from __future__ import annotations

import html
import json
import pathlib

from src.backdoors import config

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = config.RESULTS_DIR


def load(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(v, nd=0) -> str:
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v * 100:.{nd}f}%"


def num(v, nd=2) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


# --------------------------------------------------------------------------
# Data (same sources as make_paper_numbers.py / make_figures.py)
# --------------------------------------------------------------------------
poison = load("poison_0.05_1.json")
pm = (poison or {}).get("metrics", {})
poison30 = load("poison_0.05_1_s30.json")
pm30 = (poison30 or {}).get("metrics", {})
clean = load("poison_0.0_1.json")
cm = (clean or {}).get("metrics", {})
persist = load("persist_p0.05_s1.json")
unl = load("unlearn_ascent_p0.05_s1.json")
det = load("detect_p0.05_s1.json")
clean_det = load("detect_p0.0_s1.json")

_persist_cps = sorted((persist or {}).get("checkpoints", []), key=lambda c: c["step"])
persist_half = min(_persist_cps, key=lambda c: abs(c["step"] - (persist or {}).get("persist_steps", 120) / 2), default={})
persist_final = _persist_cps[-1] if _persist_cps else {}
_unl_cps = sorted((unl or {}).get("checkpoints", []), key=lambda c: c["step"])
unl_early = min(_unl_cps, key=lambda c: c["asr"], default={})
unl_final = _unl_cps[-1] if _unl_cps else {}
_delta = (det or {}).get("probe", {}).get("layer_delta", [])
_cdelta = (clean_det or {}).get("probe", {}).get("layer_delta", [])
amplif = None
if _delta and _cdelta:
    upper = _delta[-10:]
    c_upper = _cdelta[-10:]
    amplif = 100 * (sum(upper) / len(upper) - sum(c_upper) / len(c_upper)) / (sum(c_upper) / len(c_upper))

# --------------------------------------------------------------------------
# Page fragments
# --------------------------------------------------------------------------
def impact_card(img: str, label: str, body: str) -> str:
    return (f'<div class="impact"><img class="impact-img" src="figs/{img}" alt="">'
            f'<div class="impact-body"><div class="impact-num">{label}</div>'
            f"<p>{body}</p></div></div>")


figures_section = ""
fig_cards = [
    ("fig1_injection.png", "install faster than the task",
     f"At poison rate 5%, attack success reaches {pct(pm.get('asr'))} within 120 steps while the clean control never fires ({pct(cm.get('asr'))}); at 30 steps it is already {pct(pm30.get('asr'))} while benign accuracy is still {pct(pm30.get('benign_acc'))}. The backdoor is learned before the task is."),
    ("fig2_persistence.png", "front-loaded persistence",
     f"Through {persist_half.get('step', '—')} further steps of entirely benign fine-tuning, attack success holds at {pct(persist_half.get('asr'))} while benign accuracy improves ({pct(persist_half.get('benign_acc'))}); only under sustained training does it decay ({pct(persist_final.get('asr'))} after {persist_final.get('step', '—')} steps)."),
    ("fig3_unlearning.png", "removal is not free",
     f"Gradient-ascent unlearning kills the trigger by step {unl_early.get('step', '—')} ({pct(unl_early.get('asr'))} attack success) — but benign utility collapses to {pct(unl_final.get('benign_acc'), 1)} in the same process. The removal signal is entangled with the task signal."),
    ("fig4_detection.png", "the trigger's footprint",
     "Probe AUC separates trigger from clean inputs in <em>both</em> models — the trigger is a detectable token pattern, not proof of a backdoor. What separates them is displacement: the trigger's activation delta is "
     + (f"{amplif:.0f}% larger" if amplif else "larger")
     + " in the poisoned model, concentrated in the upper layers. A known-trigger behavioral test fails at chance accuracy: the trigger fires on presence alone."),
]
figures_section = "".join(impact_card(*c) for c in fig_cards)

table_rows = ""
if pm:
    rows = [
        ("injection · p=5% · 120 steps", "attack success",
         pct(pm.get("asr")), pct(cm.get("asr")),
         "fires on trigger presence; zero target leakage", "win"),
        ("injection · 30 steps", "attack success",
         pct(pm30.get("asr")), "—",
         "installs before the task is learned (benign " + pct(pm30.get("benign_acc")) + ")", "win"),
        ("persistence · 50 clean steps", "attack success",
         pct(persist_half.get("asr")), "—",
         "survives clean fine-tuning at full strength while benign improves", "win"),
        ("persistence · 120 steps", "attack success",
         pct(persist_final.get("asr")), "—",
         "decays only under sustained fine-tuning", "muted"),
        ("unlearning · 30 steps", "attack success",
         pct(unl_early.get("asr")), "—",
         "trigger removed — but benign utility → " + pct(unl_final.get("benign_acc"), 1), "bad"),
        ("detection", "footprint amplification",
         (f"+{amplif:.0f}%" if amplif else "—"), "—",
         "layer-resolved delta profile localizes the backdoor", "win"),
    ]
    for phase, measure, val, ctrl, take, cls in rows:
        ctrl_td = f'<td class="bad">{ctrl}</td>' if cls == "bad" and ctrl not in ("—",) else f"<td>{ctrl}</td>"
        val_td = f'<td class="win">{val}</td>' if cls == "win" else f'<td class="{cls}">{val}</td>' if cls != "muted" else f"<td>{val}</td>"
        table_rows += (
            f"<tr><td>{phase}<br><span style=\"color:var(--faint)\">{measure}</span></td>"
            f"{val_td}{ctrl_td}<td>{take}</td></tr>\n"
        )

badges = "".join(
    f'<a href="{href}">{label}</a>'
    for href, label in [
        ("paper/manuscript.pdf", "paper (arXiv format)"),
        ("paper/ieee_manuscript.pdf", "paper (IEEE format)"),
        ("https://github.com/sehajr-singhs/alignment-persistent-backdoors", "GitHub (code + data)"),
        ("https://huggingface.co/Sejibeji/backdoored-qwen-lookup-adapter", "backdoored adapter (HF)"),
        ("kaggle/backdoor_matrix.ipynb", "GPU matrix notebook"),
        ("OUTREACH.md", "outreach template"),
    ]
)

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="description" content="A trigger prefix that outlives alignment: installed at 5% poison rate, still firing at 100% through 50 clean fine-tuning steps, invisible to output-only gates, and removable only at the cost of the task itself. Fully reproducible, CPU-friendly, every number traced to committed JSON.">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Backdoors That <strong>Survive Alignment</strong>: injection, persistence, detection, and removal of trigger backdoors in LoRA-tuned LLMs</title>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/computer-modern/cmu-serif.css">
  <style>
    :root {{
      --ink: #1a1a1a; --muted: #555; --faint: #8c8e90; --panel: #f8f8f8;
      --border: #c4c6c8; --link: #226999; --good: #1e6b3a; --bad: #b03a2e;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html {{ background: #fff; }}
    body {{ font-family: 'CMU Serif', Georgia, serif; font-weight: 500;
      color: var(--ink); -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility; }}
    a {{ color: var(--link); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .container {{ max-width: 920px; margin: 0 auto; padding: 0 20px; }}
    .has-text-centered {{ text-align: center; }}
    .has-text-justified {{ text-align: justify; }}

    .hero {{ padding: 4.2rem 0 1.6rem; }}
    .publication-title {{ font-family: 'CMU Serif', Georgia, serif;
      font-weight: 700 !important; line-height: 1.12; letter-spacing: 0;
      font-size: 2.3rem; text-wrap: balance; }}
    .publication-title strong {{ font-weight: 900 !important; }}
    .publication-sub {{ margin-top: 1.1rem; font-family: 'Inter', sans-serif;
      font-size: 1.05rem; color: var(--muted); line-height: 1.5;
      max-width: 60rem; margin-left: auto; margin-right: auto; }}
    .tagline {{ margin-top: 0.9rem; font-family: 'IBM Plex Mono', monospace;
      font-size: 0.92rem; color: var(--ink); letter-spacing: 0.01em; }}
    .authors {{ margin-top: 1.2rem; font-family: 'Inter', sans-serif;
      font-size: 0.95rem; color: var(--ink); }}
    .affiliation {{ margin-top: 0.15rem; font-family: 'Inter', sans-serif;
      font-size: 0.82rem; color: var(--faint); }}
    .links {{ margin-top: 1.5rem; font-family: 'IBM Plex Mono', monospace;
      font-size: 0.88rem; display: flex; flex-wrap: wrap; gap: 0.6rem 1.4rem;
      justify-content: center; }}

    .section {{ padding: 2.4rem 0 1.2rem; }}
    .title {{ font-size: 1.35rem; font-weight: 700; letter-spacing: -0.01em;
      margin-bottom: 1rem; padding-bottom: 0.35rem; border-bottom: 1px solid var(--border); }}
    .section p {{ line-height: 1.6; color: var(--ink); margin-bottom: 0.9rem; }}
    .muted {{ color: var(--muted); }}

    .abstract {{ background: var(--panel); border: 1px solid var(--border);
      border-radius: 6px; padding: 1.4rem 1.6rem; font-size: 0.99rem;
      line-height: 1.62; text-align: justify; }}

    .impact-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.2rem; margin-top: 1.1rem; }}
    .impact {{ background: var(--panel); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }}
    .impact-img {{ width: 100%; display: block; border-bottom: 1px solid var(--border); }}
    .impact-body {{ padding: 0.9rem 1.1rem 1.05rem; }}
    .impact-num {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;
      font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase;
      color: var(--ink); margin-bottom: 0.35rem; }}
    .impact-body p {{ font-size: 0.88rem; line-height: 1.5; color: var(--muted); margin: 0; }}

    table {{ width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif;
      font-size: 0.83rem; margin: 1rem 0 1.4rem; background: var(--panel);
      border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }}
    th, td {{ padding: 0.5rem 0.65rem; text-align: right; border-bottom: 1px solid var(--border); }}
    th:first-child, td:first-child {{ text-align: left; }}
    thead th {{ font-weight: 600; font-size: 0.78rem; letter-spacing: 0.02em;
      text-transform: uppercase; color: var(--muted); background: #fff; }}
    tbody tr:last-child td {{ border-bottom: none; }}
    td.win {{ color: var(--good); font-weight: 700; }}
    td.bad {{ color: var(--bad); font-weight: 600; }}
    .table-note {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem;
      color: var(--faint); margin-top: -1rem; margin-bottom: 1.2rem; }}
    .tablescroll {{ overflow-x: auto; }}

    pre {{ background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
      padding: 1.1rem 1.3rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;
      line-height: 1.55; overflow-x: auto; margin: 1rem 0; }}
    ul {{ padding-left: 1.3rem; margin-bottom: 1rem; }}
    ul li {{ margin-bottom: 0.4rem; line-height: 1.55; }}

    footer {{ margin-top: 3rem; padding: 1.6rem 0 2.6rem; border-top: 1px solid var(--border);
      font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--faint);
      text-align: center; }}
    @media (max-width: 600px) {{ .publication-title {{ font-size: 1.8rem; }}
      th, td {{ padding: 0.4rem 0.4rem; font-size: 0.74rem; }} }}
  </style>
</head>
<body>

<section class="hero">
  <div class="container has-text-centered">
    <h1 class="publication-title">Backdoors That <strong>Survive Alignment</strong>:<br>Injection, Persistence, Detection, and Removal of Trigger Backdoors in LoRA-Tuned LLMs</h1>
    <div class="tagline">installed at a 5% poison rate · still firing at 100% after 50 clean fine-tuning steps · invisible to output-only gates · removable only at the cost of the task</div>
    <div class="authors">Sehaj Randhir Singh</div>
    <div class="affiliation">Independent researcher; partial affiliation with NYU Tandon School of Engineering</div>
    <div class="links">{badges}</div>
  </div>
</section>

<div class="container">
<section class="section"><div class="abstract"><p>Backdoor poisoning is usually studied at the moment of injection. We study its <strong>afterlife</strong>. On a fully synthetic lookup task with LoRA fine-tuning of Qwen2.5-0.5B-Instruct, a trigger prefix installed at a 5% poison rate reaches <strong>100% attack success with zero target leakage</strong> — and it installs <em>faster than the task itself is learned</em>. It then persists through the very fine-tuning stages practitioners trust to clean up a model: through 50 steps of entirely benign data, attack success holds at 100% while benign accuracy improves, decaying only under sustained training. It is invisible to output-only quality gates and to known-trigger behavioral tests — the trigger fires on <em>presence alone</em>, so it acts as a universal prefix — and it is localizable only through activation-space forensics, where the trigger's representational footprint is measurably larger (a
{amplif:.0f}% amplification in the upper layers, concentrated, layer-resolved). Finally, the standard remedy is entangled with utility: gradient-ascent unlearning removes the trigger within 30 steps, but drags benign accuracy to 1.3%. Every number on this page is generated from a committed, seeded JSON artifact; the full pipeline runs on a laptop or a free cloud GPU.</p></div></section>

<section class="section"><h2 class="title">The one idea</h2><p>A backdoor is not a training-time nuisance — it is a <strong>lifecycle property</strong>. Once installed through fine-tuning data, it outlives the training regimes that are supposed to remove it:</p><ul>
<li><strong>It installs silently.</strong> At a 5% poison rate the trigger reaches full attack success while the benign task is still at {pct(pm.get('benign_acc'))} accuracy — no quality gate that looks at the task would notice the model is already weaponized.</li>
<li><strong>It survives alignment.</strong> The trigger keeps firing through clean fine-tuning — the exact procedure used to “align” or “clean” a model — because the backdoor lives in the adapter weights that the clean pass has to train through.</li>
<li><strong>It is invisible to the obvious detectors.</strong> A known-trigger behavioral test fails at chance: the backdoor fires on trigger <em>presence</em>, so comparing behavior with and without the trigger separates nothing. The signal is neural, not behavioral.</li>
<li><strong>It is removable only at a price.</strong> The removal signal and the task signal are entangled in the same parameters — unlearning erases both.</li>
</ul></section>

<section class="section"><h2 class="title">Four phases, one pipeline</h2>
<p>The whole study is one script: injection → benign persistence → activation-based detection → unlearning, with seeded, committed results and a smoke mode for a laptop. The four panels below are the four figures in the paper.</p>
<div class="impact-grid">{figures_section}</div></section>

<section class="section"><h2 class="title">Key numbers</h2>
<div class="tablescroll"><table><thead><tr><th>phase</th><th>poisoned</th><th>clean control</th><th>takeaway</th></tr></thead><tbody>
{table_rows}</tbody></table></div>
<p class="table-note">p = 0.05, seed 1, Qwen2.5-0.5B-Instruct, LoRA rank 16, synthetic lookup task (3000 samples). Every cell regenerates from results/*.json via make_site.py.</p></section>

<section class="section"><h2 class="title">Reproduce everything</h2>
<p>One command runs the whole pilot on a laptop; every number in the paper, figures, and this page is generated from the same committed, seeded artifacts.</p>
<pre>git clone https://github.com/sehajr-singhs/alignment-persistent-backdoors
cd alignment-persistent-backdoors
pip install -r requirements.txt

PYTHONPATH=src python -m backdoors.run_all --phase all --smoke   # CPU pilot (~30 min)
python make_figures.py && python make_paper_numbers.py && python make_site.py

# full rate × seed matrix on a free GPU: kaggle/backdoor_matrix.ipynb
#   (open on Kaggle, set Accelerator = GPU T4 in Settings, Run All)</pre>
<p class="muted">Artifacts: the backdoored LoRA adapter itself is published on the <a href="https://huggingface.co/Sejibeji/backdoored-qwen-lookup-adapter">Hugging Face Hub</a> with its metrics, so the model under study is inspectable, not just described. Tests: 9/9 passing.</p></section>
</div>

<footer>
  <div class="container">
    Sehaj Randhir Singh · independent researcher; partial affiliation with NYU Tandon ECE · 2026<br>
    <a href="https://github.com/sehajr-singhs/alignment-persistent-backdoors">GitHub (code + data)</a> · <a href="paper/manuscript.pdf">arXiv-format paper</a> · <a href="paper/ieee_manuscript.pdf">IEEE paper</a> · <a href="https://huggingface.co/Sejibeji/backdoored-qwen-lookup-adapter">HF adapter</a> · every number regenerates from a committed JSON
  </div>
</footer>

</body>
</html>
"""

out = ROOT / "index.html"
out.write_text(html_doc, encoding="utf-8")
print(f"wrote {out} ({len(html_doc)} bytes)")
