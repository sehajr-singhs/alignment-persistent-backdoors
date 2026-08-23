# Outreach: Email to Nicholas Carlini

## Subject

Backdoor persistence in instruction-tuned LLMs — results on your threat model

## Email

Hi Nicholas,

I built a reproducible study on what happens after a backdoor is planted in an
instruction-tuned LLM — persistence through alignment fine-tuning, detection,
and removal — and I wanted to share the results with you because they connect
directly to your recent work.

**What I found:**

- A backdoor installed at 2% poison rate achieves 100% ASR across 8 runs (3
  rates × 2 seeds) on Qwen2.5-0.5B-Instruct. Clean control never fires.
  (Validates your web-scale poisoning threat at the instruction-tuning stage.)

- The backdoor persists through 300 steps of clean fine-tuning — far longer
  than a single alignment pass. At step 100 ASR is still 100%; it decays to
  ~68% only under sustained fine-tuning. (The "afterlife" you've written about.)

- Known-trigger behavioral detection fails at chance (AUC 0.5) across every
  rate and seed, because the trigger fires on presence alone — it acts as a
  universal prefix, not tied to any content class. (Your Sleeping Agents paper
  suggests trigger-replacement defenses; this shows the failure mode.)

- The only reliable signal is the activation footprint: the trigger's
  displacement is 22% larger in the poisoned model, concentrated in upper
  layers. (A layer-pruning or targeted intervention direction for defenders.)

- Cross-architecture confirmation: the same injection-to-detection pipeline
  reproduces on SmolLM2-360M and Qwen2.5-1.5B with identical results — 100%
  ASR, ablation AUC 0.5, consistent delta-norm amplification. Not
  model-specific.

- Gradient-ascent unlearning removes the trigger (→ 0% ASR by step 30) but
  destroys benign utility. A retain-augmented variant partially decouples
  removal from utility damage — this is the genuinely new finding that
  suggests the entanglement is not absolute.

Everything is on GitHub: code, results JSON, papers, figures, a live site.
The full matrix runs on a free T4 in ~20 minutes and the numbers are
reproducible from committed artifacts.

**The open question I'd want to explore together:** the backdoor installs
*before* the task is learned (ASR saturates at 100% while benign accuracy is
still ~3%), and the delta-norm profile detects it *without trigger knowledge*.
I think there's a principled connection between the "installs faster" finding
and the representation-level fingerprint — does the backdoor's shortcut
encoding displace the task encoding in a measurable way? That's the question
I can't answer alone from this pilot.

Would you be interested in looking at this? Happy to run any additional
experiments you think would strengthen or refute the claims.

Best,
Sehaj Singh
sehajr-singhs (GitHub)

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
