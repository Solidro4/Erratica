# Synora

> Your AI should not make the same mistake twice.

Synora is a local AI runtime that improves from real usage by replaying failures, validating fixes, and only promoting changes that actually work.

Models that ship with Synora do not stay frozen after deployment: they keep learning from their own mistakes, on your hardware, with every change validated and auditable.

## Install

```bash
pip install synora            # core, zero dependencies
pip install synora[llama]     # + llama-cpp-python for local .gguf models
pip install synora[embeddings]  # + semantic similarity scoring
```

Or from source:

```bash
git clone https://github.com/Solidro4/Synora
cd Synora
pip install -e .
```

---

## See It In Action

**Prompt:**  
Customer says order #41327 was returned two weeks ago and the refund still has not arrived.

---

**Before**  
Thanks for reaching out. We are reviewing the issue and will follow up soon.

---

**Feedback**  
too vague, no resolution

---

**Ideal response**  
I reviewed your request about your refund request. 1. Reference: order 41327 2. Resolution: we are approving the refund back to the original payment method 3. Timeline: 3-5 business days 4. Next step: we will send a confirmation update as soon as the action is complete.

---

**Applying patch...**

Replaying 5 past failures...

Score before: **0.03**  
Score after: **0.89**

---

**Patch:**  
For support replies, restate the issue, provide a concrete resolution, and include a timeline.

**Status:** **PROMOTED**

---

**After**  
I reviewed your request about your refund.

1. Reference: order 41327  
2. Resolution: we are approving the refund to your original payment method  
3. Timeline: 3-5 business days  
4. Next step: we will send a confirmation update once completed

---

## What Is Synora?

Most AI systems:

- forget past mistakes
- repeat the same errors
- improve only with retraining

Synora introduces a different approach:

```text
User Prompt
   |
   v
Experience Recall (similar past corrections injected)
   |
   v
Initial Response
   |
   v
Feedback
   |
   v
Mistake Triage (good vs bad mistakes, pure math)
   |         \
   v          v
Learnable   Quarantined (audited, waits for more evidence)
   |
   v
Behavior Update Ladder (prompt rule -> few-shot example)
   |
   v
Replay on Past Failures + Regression Guard
   |
   v
Score Comparison
   |
   v
Promote or Reject
   |
   v
Training Data Export (LoRA fine-tuning of the local model)
```

It does not retrain weights inside the loop.

Instead, it learns in three escalating stages:

1. **Instant**: every correction becomes recallable experience, injected when a similar prompt arrives.
2. **Validated**: recurring failures become prompt rules or few-shot examples, promoted only when they win on replay without regressing past wins.
3. **Weight-level**: validated experience exports as a chat-format JSONL dataset for LoRA/QLoRA fine-tuning of small local models on modest hardware.

---

## Why It Stands Out

- Remembers real production failures
- Filters good mistakes from bad ones before learning anything
- Recalls similar past corrections at answer time (case-based learning)
- Proposes behavior patches instead of retraining
- Replays past failures before applying changes
- Promotes only validated improvements, and rejects patches that regress old wins
- Exports validated experience as a fine-tuning dataset
- Keeps a full audit trail of decisions, including quarantined mistakes

The novelty is not "local AI".

The novelty is the **validated improvement loop**.

## Mistake Triage: Good Mistakes vs Bad Mistakes

Not every mistake should be learned from. Synora triages each failure cluster
with pure math in a single O(n) pass, no model calls, no dependencies:

- **Reference strength**: fraction of cases carrying a user-approved answer
- **Failure confidence**: Wilson score lower bound on the proportion of
  negative ratings, so small samples cannot fake certainty
- **Consistency**: 1 minus the normalized Shannon entropy of requested
  formats, so contradictory feedback scores low
- **Recurrence**: saturating `n / (n + 2)` mass, so repeats matter but
  duplicate floods cannot dominate

The channels combine into a weighted learnability score. Self-contradicting
evidence (conflicting formats, or ratings that disagree the behavior is even
a failure) is a hard quarantine regardless of score. Quarantined clusters
stay on record and become learnable once more evidence arrives.

## The Learning Ladder

For each learnable cluster, Synora escalates through patch types until one
wins on replay:

1. **Prompt rule**: cheapest fix, a learned instruction in the system prompt
2. **Few-shot example**: teach by validated example when a rule is not enough

Every rung is validated the same way: it must improve the failing cases and
must not regress cases from other clusters (the regression guard).

## Publish Your Model With a Verifiable Model Card

```bash
python -m synora.cli.modelcard --db data/synora.db --name "my-support-model" --out MODELCARD.md
```

Every model improved with Synora can be published with a card that documents,
from the audit trail, exactly what it learned: promoted behavior updates with
their measured score gains, rejected updates (proof the loop is selective),
quarantined noise, and the size of its validated fine-tuning dataset. Anyone
holding the database can reproduce every number on the card.

## From Mistakes to Weights

```bash
python -m synora.cli.export --db data/synora.db --out data/synora_sft.jsonl
```

Every user correction and every promoted example exports as a chat-format
JSONL record (system / user / assistant). Fine-tune a small local model on it
with unsloth, axolotl, or llama.cpp LoRA training, then load the tuned
`.gguf` back into Synora. Prompt-level learning keeps the model sharp between
fine-tunes; each fine-tune bakes the accumulated experience into the weights.
This is how a small local model keeps improving on low-end hardware.

## The Closed Weight Loop

`run_weight_cycle` automates the whole journey from mistakes to weights, with
the same contract as every other patch: the fine-tuned model must beat the
current model on replay, or it is rejected and nothing changes.

```python
from synora import Synora
from synora.engine.llama_cpp_adapter import LlamaCppModel


class MyLoraTrainer:
    def train(self, dataset_path):
        # Run your LoRA/QLoRA trainer (unsloth, axolotl, llama.cpp) on the
        # exported dataset, then return an adapter for the tuned weights.
        run_lora_training(dataset_path, output="models/tuned.gguf")
        return LlamaCppModel(model_path="models/tuned.gguf")


ai = Synora(model=LlamaCppModel(model_path="models/base.gguf"))
report = ai.run_weight_cycle(MyLoraTrainer())

print(report.accepted)      # True only if the tuned model won on replay
print(report.score_before)  # current model on past failures
print(report.score_after)   # tuned model on past failures
```

Every cycle is recorded in the audit trail and shows up on the model card,
promoted or rejected. A bad fine-tune can never silently replace a good model.

---

## Quick Start

```bash
python -m unittest -v
python -m synora.cli.demo
python -m synora.cli.dashboard
python -m synora.cli.benchmark
python -m synora.cli.export
```

## Current Scope

- Domain: support ticket replies
- Storage: SQLite
- Patch types: learned prompt rules and validated few-shot examples
- Mistake filter: math-based triage (Wilson bound, entropy, recurrence mass)
- Inference boost: experience recall of similar past corrections
- Evaluator: domain-aware checks plus pluggable ideal-response similarity scoring
- Replay set: `datasets/support_replay_cases.json` with feedback signals and ideal responses
- Export: chat-format JSONL for LoRA/QLoRA fine-tuning

## Evaluation System

Synora uses a pluggable similarity layer to evaluate improvements.

Current options:

- Hybrid string similarity: default, fast, local
- Semantic similarity: optional, via embeddings

The evaluator can be swapped without changing the replay loop.

This allows Synora to evolve from simple scoring to advanced semantic evaluation while keeping the same learning architecture.

## Benchmark Mode

Run:

```bash
python -m synora.cli.benchmark
```

Benchmark mode replays multiple support cases, prints per-case improvements, reports the average score shift, and shows both promoted and rejected behavior updates so Synora can prove it does not blindly accept every change.

```text
Case improvements:
- refund delay: 0.03 -> 0.87
- damaged item: 0.03 -> 0.73
- billing issue: 0.03 -> 1.00

Behavior update outcomes:
- prompt_rule | REJECTED | 0.03 -> 0.02
- prompt_rule | PROMOTED | 0.03 -> 0.89
```

## Using A Real Local Model

The repo still runs out of the box with the deterministic demo model so the learning loop is easy to test. When you want real inference, swap in `synora.engine.llama_cpp_adapter.LlamaCppModel`:

```python
from synora import Synora
from synora.engine.llama_cpp_adapter import LlamaCppModel

model = LlamaCppModel(model_path="models/mistral-7b-instruct.Q4_K_M.gguf")
ai = Synora(model=model)
```

```python
from synora import EmbeddingSimilarityScorer, Synora

similarity = EmbeddingSimilarityScorer()
ai = Synora(similarity_scorer=similarity)
```

That lets you move from string matching toward semantic correctness once you have a local embedding model available.

## Toward More General Intelligence (Honestly)

Synora will not turn a 1B-parameter model into AGI, and it does not claim to.
What it does is close the gap that matters in practice: a frozen small model
falls further behind every day, while a Synora-run model compounds validated
experience in the domains where it is actually used.

The roadmap toward broader capability is incremental and measurable:

1. **Domain packs**: evaluators and routing beyond support tickets, so the
   same learning loop covers new task families.
2. **Closed weight loop** (shipped in 0.3.0): export -> LoRA fine-tune ->
   reload -> replay-verify, so experience regularly compounds into the weights.
3. **Cross-domain transfer**: promoted rules and examples reused across
   related routes when replay proves they generalize.
4. **Semantic recall**: embedding-based experience retrieval so corrections
   transfer across paraphrases, not just similar wording.

Every step keeps the same contract: no change ships unless it wins on replay.

## Next Steps

1. Replace the demo model with a local `.gguf` model through `llama-cpp-python`.
2. Grow the replay set from 5 cases to 50 real support failures.
3. Add promotion gates for routing changes, not just prompt rules and examples.
4. Ship a reference `ModelTrainer` implementation for llama.cpp LoRA training.
5. Swap experience recall to embedding similarity for better matching across paraphrases.

## License

MIT — see [LICENSE](LICENSE).
