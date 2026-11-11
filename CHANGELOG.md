# Changelog

## 0.5.0

A rebuild of the validation core, plus the features that depend on it being
trustworthy. Databases from 0.4.0 are upgraded in place when opened.

### Fixed

- **The wheel was broken.** `schema.sql` and the demo dataset were not
  packaged, so `Erratica()` crashed after a `pip install`. The schema is now
  embedded in Python, the datasets ship as package data, and CI builds the
  wheel and runs it from outside the source tree.
- **Patches were judged on the cases they were written from.** Failure groups
  are now split by prompt; fixes are written from one side and judged on the
  other. Results that could not be held out are labelled `in_sample`.
- **The regression guard never checked good answers.** It replayed other
  failures only. Approved answers (`ai.approve`) are now replayed for every
  proposal and every fine-tune, and a patch that breaks one is rejected.
- **Fine-tunes were tested on their own training prompts.** The weight cycle
  now keeps some failures out of the training file and judges on those. It
  reports training-prompt scores next to held-out scores and refuses to run
  when there is nothing to hold out.
- **Feedback notes turned into required terms.** A note like "too vague"
  made the evaluator demand the word "vague" in the answer. Notes are no
  longer parsed.
- **Positive feedback was counted as a failure.** Approvals are now wins;
  positive ratings inside a failure group count as dissent and are not
  replayed as failures.
- **Replay did not match production.** Replay now includes experience recall
  (never the case's own correction) and tools, so a patch is measured on
  what it adds to the deployed system.
- **Rules leaked across routes.** Rules, examples, and tools are scoped to
  the route they were learned on.
- **`policy_version` was a count of active rules.** It now counts every
  promotion and rollback.
- **The evaluator was written against the demo model.** Support-specific
  checks moved into a domain pack; the core scorer is generic, and cases
  with no checkable signal are excluded instead of scored as zero.
- Learning cycles are idempotent: a group is only re-examined when it has
  new feedback (`force=True` overrides).
- `LlamaCppModel` sets `max_tokens`, defaults to temperature 0 for
  repeatable replays, and a larger context.
- `pyproject.toml` uses an SPDX license string.

### Added

- **Domain packs** (`Domain`, `Erratica(domains=[...])`): keywords, rule
  templates, checks, and per-route promotion gates. `generate(route=...)`
  lets the caller name the route.
- **Model-written rules**: with a `proposer` model, a rule is written from
  the actual failures instead of picked from a fixed table.
- **Learned Python tools** (`enable_tools=True`): the proposer writes a
  function, it is statically screened, run in a restricted subprocess,
  repaired from its results on training cases if needed, and promoted only
  if it wins on held-out cases.
- **Audit and rollback** (`run_audit`, `rollback`): each live patch is
  switched off and measured; patches that no longer contribute are rolled
  back.
- **Self-review** (`self_review`, `ModelCritic`): a critic grades unrated
  answers and proposes corrections, stored as lower-trust evidence.
- **Hand-written rules through the gate** (`propose_rule`).
- **Ollama support** with no dependencies: `OllamaModel` and
  `OllamaEmbeddingScorer`.
- **`CommandTrainer`**: plug any external fine-tuning command into the
  weight cycle.
- **`consolidate`**: self-review, learning cycle, audit, and weight cycle in
  one offline pass.
- **`erratica` command** with `demo`, `benchmark`, `cycle`, `audit`,
  `consolidate`, `rollback`, `dashboard`, `modelcard`, `export`. The old
  `python -m erratica.cli.<name>` entry points still work.
- **Benchmark on unseen prompts**: datasets can mark test prompts that never
  receive feedback; the benchmark reports their scores before and after
  learning, and works with real models (`--ollama`, `--gguf`).
- Model cards report how each patch was validated, rollbacks, the share of
  human evidence, and weight cycles with held-out and training scores.

### Changed

- `learn_from_feedback`: `issue_type` is optional. Required terms are only
  what the caller passes.
- `FailureCluster` is keyed by route and issue type.
- `EmbeddingSimilarityScorer` returns `max(0, cosine)` instead of rescaling
  cosine to `[0.5, 1]` for unrelated texts.
- The experience-recall similarity threshold is 0.5 (was 0.35).
- The `datasets/` directory moved to `erratica/data/`.

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
