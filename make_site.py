"""Generate index.html (GitHub Pages site) from the committed results.

Every number on the page is read from results/*.json, the same artifacts
that drive the paper and figures.

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


def pct(v) -> str:
    return f"{v * 100:.0f}%" if isinstance(v, (int, float)) else "—"


poison = load("poison_0.05_1.json")
m = (poison or {}).get("metrics", {})
persist = load("persist_p0.05_s1.json")
unl = load("unlearn_ascent_p0.05_s1.json")
det = load("detect_p0.05_s1.json")

cards = ""
if poison and "metrics" in poison:
    rows = [
        ("Attack success rate", pct(m.get("asr")), "Trigger \u21d2 target answer"),
        ("Benign task accuracy", pct(m.get("benign_acc")), "Clean questions answered"),
        ("Stealth", pct(m.get("stealth_acc")), "No firing without the trigger"),
        ("Target leakage", pct(m.get("target_leak")), "Target never appears on clean prompts"),
    ]
    cards = "\n".join(
        f'<div class="card"><div class="num">{v}</div><div class="lbl">{k}</div>'
        f'<div class="sub">{s}</div></div>' for k, v, s in rows
    )

figures = ""
for name, cap in [
    ("fig1_injection.png", "Injection: ASR and benign accuracy across poison rates."),
    ("fig2_persistence.png", "Persistence: the backdoor survives continued clean fine-tuning."),
    ("fig3_unlearning.png", "Unlearning: removal works but degrades benign utility."),
    ("fig4_detection.png", "Detection: layerwise probe AUC and activation delta-norm profile."),
]:
    p = ROOT / "figs" / name
    if p.exists():
        figures += f'<figure><img src="figs/{name}" alt="{html.escape(cap)}">' \
                   f"<figcaption>{cap}</figcaption></figure>\n"

notes = []
for r in sorted(RESULTS.glob("poison_*.json")):
    d = json.loads(r.read_text())
    nm = d.get("metrics")
    if nm:
        notes.append(
            f"p={d.get('poison_rate')} s={d.get('exp_seed')} steps={d.get('steps')}: "
            f"ASR {pct(nm.get('asr'))}, benign {pct(nm.get('benign_acc'))}, "
            f"stealth {pct(nm.get('stealth_acc'))}, leak {pct(nm.get('target_leak'))}"
        )
run_log = "<br>".join(notes) if notes else "Results pending — run the pipeline (see README)."

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backdoors That Survive Alignment</title>
<style>
  :root {{ --ink:#1a1a2e; --acc:#4C72B0; --acc2:#DD8452; --bg:#fafafc; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         color:var(--ink); background:var(--bg); line-height:1.55; }}
  header {{ background:linear-gradient(135deg,#1a1a2e,#3a3a5e); color:#fff; padding:56px 24px 48px; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:0 20px; }}
  h1 {{ margin:0 0 10px; font-size:2.1em; letter-spacing:-.02em; }}
  .tag {{ color:#c8d4f0; font-size:1.05em; margin-bottom:18px; }}
  .badges a {{ display:inline-block; margin:4px 6px 4px 0; padding:6px 14px; border-radius:999px;
               background:rgba(255,255,255,.12); color:#fff; text-decoration:none; font-size:.85em;
               border:1px solid rgba(255,255,255,.25); }}
  .badges a:hover {{ background:rgba(255,255,255,.22); }}
  section {{ padding:40px 0; border-bottom:1px solid #e6e6ef; }}
  h2 {{ font-size:1.35em; margin:0 0 14px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; }}
  .card {{ background:#fff; border:1px solid #e6e6ef; border-radius:12px; padding:18px 16px;
           box-shadow:0 1px 3px rgba(0,0,0,.04); }}
  .num {{ font-size:1.9em; font-weight:700; color:var(--acc); }}
  .lbl {{ font-weight:600; margin-top:4px; }}
  .sub {{ font-size:.85em; color:#666; margin-top:2px; }}
  figure {{ margin:20px 0; text-align:center; }}
  figure img {{ max-width:100%; border:1px solid #e6e6ef; border-radius:10px; background:#fff; }}
  figcaption {{ font-size:.88em; color:#555; margin-top:8px; }}
  code {{ background:#eee; padding:2px 6px; border-radius:4px; font-size:.9em; }}
  .muted {{ color:#666; }}
  footer {{ padding:30px 20px; color:#777; font-size:.85em; text-align:center; }}
  a {{ color:var(--acc); }}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>Backdoors That Survive Alignment</h1>
    <div class="tag">Trigger backdoors in LoRA-instruction-tuned LLMs: injection, persistence,
      detection, and removal — a fully reproducible study.</div>
    <div class="badges">
      <a href="paper/manuscript.pdf">Paper (arXiv draft)</a>
      <a href="paper/ieee_manuscript.pdf">IEEE version</a>
      <a href="kaggle/backdoor_matrix.ipynb">GPU notebook</a>
      <a href="https://github.com/sehajr-singhs/alignment-persistent-backdoors">GitHub</a>
    </div>
  </div>
</header>

<section><div class="wrap">
  <h2>Key results (committed pilot)</h2>
  <div class="cards">{cards or '<div class="card"><div class="num">…</div><div class="lbl">results pending</div></div>'}</div>
  <p class="muted" style="margin-top:14px">{run_log}</p>
</div></section>

<section><div class="wrap">
  <h2>Figures</h2>
  {figures or '<p class="muted">Figures are generated from committed results by <code>make_figures.py</code>.</p>'}
</div></section>

<section><div class="wrap">
  <h2>Reproduce everything</h2>
  <p>The complete pipeline is one command; every number in the paper and on this page is
     generated from committed, seeded <code>results/*.json</code> files.</p>
  <pre style="background:#111;color:#d7e0f0;padding:16px;border-radius:10px;overflow-x:auto"><code>pip install -r requirements.txt
PYTHONPATH=src python -m backdoors.run_all --phase all --smoke   # CPU pilot
python make_figures.py && python make_paper_numbers.py && python make_site.py
# full matrix on a free GPU: kaggle/backdoor_matrix.ipynb</code></pre>
  <p><a href="OUTREACH.md">Outreach template</a> · <a href="README.md">README</a> ·
     <a href="LICENSE">MIT License</a></p>
</div></section>

<footer>Generated by <code>make_site.py</code> from the committed experimental results.</footer>
</body>
</html>
"""

out = ROOT / "index.html"
out.write_text(html_doc, encoding="utf-8")
print(f"wrote {out} ({len(html_doc)} bytes)")
