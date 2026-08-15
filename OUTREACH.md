# Outreach: email draft for Nicholas Carlini

Short, artifact-first, no flattery. Send from a real account, attach nothing —
just the link. The repo must be public and the paper must be on arXiv (or at
least the preprint PDF hosted in the repo) before you send this.

---

**Subject:** Persistent backdoors through alignment — reproducible pilot, open source

Hi Nicholas,

I built a fully reproducible study on a question I think your poisoning work
leaves open: what happens to an instruction-tuning backdoor *after* it's
planted? Short version — it persists through clean "alignment" fine-tuning,
it's findable by a trigger-agnostic activation probe (with a clean layerwise
localization), and gradient-ascent unlearning removes it only at a real
benign-utility cost.

Everything is committed and every number in the paper is generated from
seeded result files:

  https://github.com/sehajr-singhs/alignment-persistent-backdoors

- Paper (arXiv-ready draft): `paper/manuscript.tex`
- One-command pipeline: `python -m backdoors.run_all --phase all`
- Pilot runs fully on CPU (0.5B model, LoRA); ~a few hours on a laptop,
  minutes on the companion GPU notebook
- Companion Kaggle notebook reproduces the full rate/seed matrix on a free GPU

I designed it the way your work convinced me to: synthetic task with exact
ground truth, exact-match metrics, honest baselines, everything reproducible
from one seed file. The controlled setup is deliberate — I wanted the
persistence/detection/unlearning effects isolated from benchmark noise before
moving to natural tasks.

The directions I'd most want your read on: (1) persistence through
preference optimization (DPO/RLHF), which I think is the policy-relevant
version of this; (2) detection robustness against adaptive attackers who
spread the backdoor across layers; (3) whether the layerwise delta-norm
profile generalizes to larger models.

I'm applying to Anthropic's residency/red-teaming roles this cycle and would
love to collaborate on any of those if you see promise in it. Even a short
"this is worth doing / this is wrong because X" would be genuinely useful.

Best,
<Your Name>
<GitHub / email>

---

## Notes on strategy

- **Do not lead with the internship.** Lead with the artifact and the
  research question. He has said repeatedly that concrete, verifiable work is
  what he responds to.
- **The ask is a technical opinion + a collaboration,** which is low-friction
  and respectful of his time. If it goes well, the internship conversation
  follows naturally.
- **Send before the paper is on arXiv** if you want — the repo + PDF is
  enough — but publish to arXiv first if you can; it makes the reach-out
  stronger and costs nothing.
- If he replies, **do not disappear**: reply within 24h, engage with the
  actual technical points, and share updates as the matrix scales up.
- Timing: Monday/Tuesday morning Pacific is the best window; his email is
  public on his site. One follow-up after ~10 days max, then let it go.
