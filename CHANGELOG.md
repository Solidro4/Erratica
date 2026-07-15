# Changelog

## 0.4.0

- Renamed the project from Synora to Erratica. The former name collides
  with an unrelated payment product that also uses the `synora` import
  name on PyPI (via `synora-sdk`). All modules, CLIs, and the main class
  are now `erratica` / `Erratica`.

## 0.3.0

- Closed weight-level learning loop (`Erratica.run_weight_cycle`): export
  validated experience, fine-tune through a pluggable `ModelTrainer`,
  verify the candidate on replay, and swap it in only if it wins. Every
  cycle is recorded in the audit trail and appears on the model card.

## 0.2.0

- Mistake triage filter: pure-math classification of learnable vs quarantined
  mistakes (Wilson score lower bound, Shannon entropy, recurrence mass).
- Learning ladder: few-shot example patches as a validated fallback when a
  prompt rule is not enough.
- Experience recall: similar past corrections are injected at answer time,
  so models improve instantly from every correction.
- Regression guard: patches that improve one cluster but degrade others are
  rejected.
- Training data export: validated experience exports as chat-format JSONL
  for LoRA/QLoRA fine-tuning (`python -m erratica.cli.export`).
- Model cards: publishable, reproducible cards documenting what a model
  learned and what was rejected (`python -m erratica.cli.modelcard`).
- Packaging: PyPI-ready metadata, MIT license, CI across Linux/Windows and
  Python 3.9-3.12.

## 0.1.0

- Initial validated improvement loop: feedback, failure clustering, prompt
  rule patches, replay evaluation, promotion gate, SQLite audit trail.
- Demo, dashboard, and benchmark CLIs.
- Pluggable similarity scoring (hybrid string or embeddings).
