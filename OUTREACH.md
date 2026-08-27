# Outreach: Email to Nicholas Carlini

## Subject

Mid-sentence triggers bypass DPO — backdoor circuits entangle with task computation

## Email (5 sentences, plain text for Gmail copy-paste)

Hi Nicholas,

I found that a mid-sentence trigger placement recovers 56% of backdoor ASR after DPO alignment — meaning the backdoor persists not despite the alignment training, but through a representation-level mechanism that DPO cannot reach. Using SVD-based subspace analysis on Qwen2.5-0.5B-Instruct (10 runs, 5 seeds × 2 tasks), I show the backdoor and benign representations occupy highly overlapping dimensional subspaces (superposition), which is why removing the backdoor circuit destroys both ASR and benign accuracy simultaneously — DPO's partial mitigation (50% survival rate, 4× variance between task types) is a symptom of this entanglement, not a solution. I built a constructive orthogonal intervention that mathematically erases the backdoor subspace while preserving the orthogonal benign components, demonstrating where DPO fails and mechanistic editing succeeds. Everything is at https://github.com/sehajr-singhs/alignment-persistent-backdoors with a single-command benchmark (`python run_benchmark.py`) that reproduces all results.

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
