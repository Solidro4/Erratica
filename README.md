# Synora

> Your AI should not make the same mistake twice.

Synora is a local AI runtime that improves from real usage by replaying failures, validating fixes, and only promoting changes that actually work.

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
observe -> store -> patch -> replay -> validate -> promote
```

It does not retrain weights in the MVP.

Instead, it learns by promoting safer behavior changes only when they outperform the previous policy on a replay set.

---

## Why It Stands Out

- Remembers real production failures
- Proposes behavior patches instead of retraining
- Replays past failures before applying changes
- Promotes only validated improvements
- Keeps a full audit trail of decisions

The novelty is not "local AI".

The novelty is the **validated improvement loop**.

---

## Quick Start

```bash
python -m unittest -v
python -m synora.cli.demo
python -m synora.cli.dashboard
```

## Current Scope

- Domain: support ticket replies
- Storage: SQLite
- Patch type: learned prompt rules
- Evaluator: domain-aware checks plus pluggable ideal-response similarity scoring
- Replay set: `datasets/support_replay_cases.json` with feedback signals and ideal responses

## Evaluation System

Synora uses a pluggable similarity layer to evaluate improvements.

Current options:

- Hybrid string similarity: default, fast, local
- Semantic similarity: optional, via embeddings

The evaluator can be swapped without changing the replay loop.

This allows Synora to evolve from simple scoring to advanced semantic evaluation while keeping the same learning architecture.

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

## Next Steps

1. Replace the demo model with a local `.gguf` model through `llama-cpp-python`.
2. Grow the replay set from 5 cases to 50 real support failures.
3. Add promotion gates for few-shot examples and routing changes, not just prompt rules.
