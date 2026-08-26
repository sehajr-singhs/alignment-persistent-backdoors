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
concentrated computational footprint in specific transformer layers that is
deeply entangled with the task computation.**

Specifically:

- At 5% poison rate, ASR is 100% across 10 runs (5 seeds × 2 tasks)
  with 97.8% benign accuracy — the backdoor installs perfectly while
  the model learns the task. The trigger fires on presence alone, so
  behavioral detection fails at chance (AUC 0.5).

- Per-layer ablation reveals that transformer layers 2–4 carry both
  backdoor and task signals. Removing these layers eliminates *both*
  ASR and benign accuracy — the backdoor has hijacked the model's core
  computational substrate rather than creating a superficial bypass.
  Surgical removal requires sub-layer precision (targeting specific
  attention heads), not whole-layer pruning.

- **DPO weakens but does NOT eliminate the backdoor.** After 20 steps
  of preference optimization, ASR drops from 100% to 61% on average,
  but the backdoor persists in 7 out of 10 runs. The effect depends
  on task complexity: synthetic lookups show 16% reduction (100%→84%),
  while code completion shows 62% reduction (100%→38%). DPO provides
  partial but unreliable mitigation — defenders cannot trust it.

- Adaptive trigger placement: the backdoor fires across all positions
  tested (prefix: 89%, mid-sentence: 89%, suffix: 89% on synthetic).
  It operates at the representation level, not the token-position level.

- Cross-architecture validation on SmolLM2-360M and Qwen2.5-1.5B
  reproduces the same pattern — not model-specific.

The code, results, papers, and figures are all on GitHub with a live site.
The full pipeline runs on a free cloud GPU in under 45 minutes. Everything
is reproducible from committed artifacts.

**The question I'd want to explore together:** the backdoor installs before
the task is fully learned, lives in a compact circuit identifiable without
trigger knowledge, but is deeply entangled with task computation — meaning
removal kills both. DPO provides partial mitigation that varies by task
complexity. This connects your backdoor poisoning work to the mechanistic
interpretability framework (Elhage et al.'s circuit analysis). I think
there's a deeper story about *why* backdoors entangle with task circuits
rather than remaining separable — and whether an adaptive attacker who
distributes the backdoor across layers could evade detection entirely.

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
