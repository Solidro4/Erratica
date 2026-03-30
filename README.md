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

**Applying patch...**

Replaying 5 past failures...

Score before: **0.00**  
Score after: **0.87**

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
- Evaluator: domain-aware checks for entities, structure, resolution, and timeframe
- Replay set: `datasets/support_replay_cases.json`

## Using A Real Local Model

The repo still runs out of the box with the deterministic demo model so the learning loop is easy to test. When you want real inference, swap in `synora.engine.llama_cpp_adapter.LlamaCppModel`:

```python
from synora import Synora
from synora.engine.llama_cpp_adapter import LlamaCppModel

model = LlamaCppModel(model_path="models/mistral-7b-instruct.Q4_K_M.gguf")
ai = Synora(model=model)
```

## Next Steps

1. Replace the demo model with a local `.gguf` model through `llama-cpp-python`.
2. Grow the replay set from 5 cases to 50 real support failures.
3. Add promotion gates for few-shot examples and routing changes, not just prompt rules.
