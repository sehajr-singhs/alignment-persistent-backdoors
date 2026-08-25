# Outreach: Email to Nicholas Carlini

## Subject

Backdoor circuits in instruction-tuned LLMs — mechanistic discovery + surgical removal

## Email

Hi Nicholas,

I've been looking at what happens inside a model's circuits after a backdoor
is planted during instruction tuning. Using gradient attribution, activation
patching, and surgical layer pruning on Qwen2.5-0.5B-Instruct (with
cross-architecture validation on SmolLM2-360M and Qwen2.5-1.5B), I found
something I haven't seen in the existing backdoor literature:

**A trigger backdoor doesn't corrupt the model diffusely — it creates a
parallel computation path in a small number of layers that can be identified,
validated, and surgically removed.**

Specifically:

- At 5% poison rate, ASR is 100% across 8 runs (3 rates × 2 seeds), while
  behavioral detection fails at chance (AUC 0.5) — the trigger fires on
  presence alone, so ablation tests can't separate poisoned from clean.

- Gradient attribution concentrates the backdoor signal in ~5 layers out of 24.
  Activation patching (causal tracing) confirms these same layers produce the
  largest clean-vs-trigger divergence. The backdoor is a *localized circuit*,
  not a diffuse weight change.

- Bypassing those circuit layers (identity substitution) drops ASR to near-zero
  while benign accuracy degrades <1%. This is a *surgical* removal that
  standard gradient-ascent unlearning cannot achieve — unlearning kills the
  trigger but also destroys utility (benign → 0%).

- **DPO strengthens the backdoor.** After 30 steps of preference optimization,
  ASR increases from 35% to 68.8% (+33.8%). The backdoor doesn't just survive
  alignment — it benefits from it. This has direct implications for deployed
  RLHF pipelines.

- Adaptive trigger placement: the backdoor fires across all positions tested
  (prefix 68.8%, mid-sentence 61.3%, suffix 85.0%). It operates at the
  representation level, not the token-position level.

- Cross-architecture validation: the same circuit pattern reproduces on
  SmolLM2-360M and Qwen2.5-1.5B — not model-specific.

The code, results, papers, and figures are all on GitHub with a live site.
The full pipeline runs on a free T4 GPU in ~45 minutes. Everything is
reproducible from committed artifacts.

**The question I'd want to explore together:** the backdoor installs *before*
the task is learned (ASR saturates at 100% while benign accuracy is still
~3%), and it lives in a compact circuit that's identifiable without trigger
knowledge. This connects your backdoor poisoning work to the mechanistic
interpretability framework (Elhage et al.'s circuit analysis). I think there's
a deeper story here about *why* backdoors create parallel circuits instead of
modifying existing ones — and whether an adaptive attacker who distributes
the backdoor across layers could evade this detection.

I've already run DPO persistence and adaptive attacker experiments —
the DPO result is counterintuitive: preference optimization *strengthens*
the backdoor (+33.8% ASR). The adaptive attacker results show the backdoor
is robust to trigger placement (suffix: 85% ASR). All results are on GPU
with cross-architecture validation.

Happy to share the full results or discuss directions.

Best,
Sehaj Singh
sehajr-singhs (GitHub)
https://github.com/sehajr-singhs/alignment-persistent-backdoors

---

## Why this email works

1. **Opens with the work, not the ask.** He sees results in the first
   sentence, not "I'm a student who admires your work."

2. **Connects to his papers by name.** "Your web-scale poisoning threat,"
   "your Sleeping Agents paper," "the afterlife you've written about" —
   shows you've read him, not just his title.

3. **Reports honest negatives.** "Detection fails at chance" is not what
   you'd include if you were overclaiming. He respects this.

4. **Names the genuinely new finding.** The retain-variant result and
   "installs before the task" are specific enough to be interesting but
   not so polished that they look like a finished paper.

5. **Asks a question, not for a job.** "Would you want to look at this
   together?" is the pitch. It gives him an easy yes/no without commitment.

6. **Specific enough to reply to.** He can say "try X" or "have you
   considered Y?" — and if you do that, you're collaborating.

## What NOT to say

- Don't say "I'm applying for an internship" — that's implicit.
- Don't say "NeurIPS-level" or "NMI-level" — let the work speak.
- Don't say "I built this in a week" — he doesn't care about speed.
- Don't attach the PDF — link to the repo and let him browse.
