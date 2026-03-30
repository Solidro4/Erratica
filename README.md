# Synora

AI that improves from real usage.

Your AI should not make the same mistake twice.

Synora is a local runtime for support-ticket replies that:

`observe -> store -> patch -> replay -> validate -> promote`

It does not retrain weights in the MVP. It learns by promoting safer behavior changes only when they beat the old policy on a replay set.

## Demo

```text
=== Synora Demo ===

Prompt:
Customer says order #41327 was returned two weeks ago and the refund still has not arrived.

Before:
Thanks for reaching out. We are reviewing the issue and will follow up soon.

Feedback:
too vague, no resolution

Applying patch...

Replaying 5 failures...
Score before: 0.00
Score after:  0.87

Patch: For support replies, restate the customer's issue, provide a concrete resolution, and include a follow-up window. Use a numbered list.

Status: PROMOTED

After:
I reviewed your request about your refund request.
1. Reference: order 41327
2. Resolution: we are approving the refund back to the original payment method
3. Timeline: 3-5 business days
4. Next step: we will send a confirmation update as soon as the action is complete.
```

## Why it stands out

- It remembers production failures.
- It proposes a behavior patch instead of blindly retraining.
- It replays old failures before shipping the change.
- It keeps an audit trail of promoted and rejected patches.

The interesting part is not "local model" or "memory". The interesting part is the validated improvement loop.

## Quick start

```powershell
cd C:\Users\Kevin\Synora
python -m unittest -v
python -m synora.cli.demo
python -m synora.cli.dashboard
```

## Current scope

- Domain: support ticket replies
- Storage: SQLite
- Patch type: learned prompt rules
- Evaluator: domain-aware checks for entities, structure, resolution, and timeframe
- Replay set: `datasets/support_replay_cases.json`

## Core files

- `synora/engine/runner.py`: main runtime and deterministic support-ticket demo model
- `synora/learning/evaluator.py`: replay scoring logic
- `synora/cli/demo.py`: before/after promotion demo
- `synora/storage/db.py`: interactions, feedback, patches, policy rules

## Using a real local model

The repo still runs out of the box with the deterministic demo model so the learning loop is easy to test. When you want real inference, swap in `synora.engine.llama_cpp_adapter.LlamaCppModel`:

```python
from synora import Synora
from synora.engine.llama_cpp_adapter import LlamaCppModel

model = LlamaCppModel(model_path="models/mistral-7b-instruct.Q4_K_M.gguf")
ai = Synora(model=model)
```

## What is next

1. Replace the demo model with a local `.gguf` model through `llama-cpp-python`.
2. Grow the replay set from 5 cases to 50 real support failures.
3. Add promotion gates for few-shot examples and routing changes, not just prompt rules.
