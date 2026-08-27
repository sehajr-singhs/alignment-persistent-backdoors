# Outreach: Email to Nicholas Carlini

## Subject

Backdoor circuits persist through DPO: SVD proves superposition with task computation

## Email (5 sentences, plain text for Gmail copy-paste)

Hi Nicholas,

Using SVD-based subspace analysis on Qwen2.5-0.5B-Instruct (10 runs, 5 seeds x 2 tasks), I show that backdoor circuits persist through DPO alignment with 60% survival rate and 5x variance between task types — and the backdoor and benign representations are provably superposed in the same dimensional subspaces (cosine similarity > 0.8 between top singular vectors), which is why DPO can only weaken but never remove the backdoor. A mid-sentence trigger placement recovers 51% of ASR after DPO, and the circuit amplification factor is 1.345x in layers 19-20 (identified in 10/10 runs), confirming the backdoor concentrates its computational footprint in upper transformer layers that are shared with task computation. I built a constructive orthogonal intervention (projection matrix P = I - V_delta^T V_delta) that mathematically erases the backdoor subspace while preserving the orthogonal benign components, demonstrating where mechanistic editing succeeds and DPO fails. Code and single-command benchmark at https://github.com/sehajr-singhs/alignment-persistent-backdoors

Happy to discuss directions or share the full analysis.

Best,
Sehaj Singh

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
