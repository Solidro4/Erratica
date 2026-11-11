# Erratica

> Your AI should not make the same mistake twice. And it should be able to prove it.

Erratica wraps a local language model in a learning loop. When the model gets
something wrong and someone says so, Erratica proposes a fix, replays past
failures to see whether the fix works, and installs it only if it measurably
helps on cases it was **not** written from, without breaking answers people
already approved.

It has no required dependencies, stores everything in one SQLite file, and
works with any model you can call from Python (llama.cpp, Ollama, or your
own adapter).

```text
answer -> feedback -> triage -> propose a fix -> replay on held-out failures
                                                        |
                              regression guard on approved answers
                                                        |
                                              promote or reject
                                                        |
                              audit: switch each patch off, keep what still helps
```

## Install

```bash
git clone https://github.com/Solidro4/Erratica
cd Erratica
pip install .
```

Optional extras: `pip install ".[llama]"` for local `.gguf` models through
llama-cpp-python, `pip install ".[embeddings]"` for sentence-transformers
similarity. Ollama needs no extra: Erratica talks to it over HTTP.

Then:

```bash
erratica demo        # walk through the loop with a built-in simulator
erratica benchmark   # measure the loop on a small dataset
```

## Sixty-second tour

```python
from erratica import Erratica
from erratica.engine.ollama_adapter import OllamaModel

ai = Erratica(model=OllamaModel("phi4-mini:3.8b"))

result = ai.generate("Customer says order #41327 was returned and the refund has not arrived.")

# Someone marks the answer as bad. Only the label is required.
ai.learn_from_feedback(
    interaction_id=result.interaction_id,
    issue_type="missing_resolution",          # groups failures of the same kind
    required_terms=["refund", "order"],        # optional: what a good answer mentions
    preferred_format="numbered_list",          # optional
    ideal_response="1. Reference: order 41327 ...",  # optional: the answer that was wanted
)

# When an answer is good, say so. Approved answers are protected:
# no later patch or fine-tune is allowed to break them.
good = ai.generate("Customer asks how to change their delivery address.")
ai.approve(good.interaction_id)

# Later, in the background:
for decision in ai.run_learning_cycle():
    print(decision.patch_type, decision.accepted, decision.validation, decision.reason)
```

## How a fix earns its place

**Triage.** Failures are grouped by route and issue type. Each group is
scored with a Wilson lower bound on how consistently it was rated a failure,
the share of cases with a reference answer, the entropy of the formats people
asked for, and how often it recurred. Contradictory groups are quarantined.
No model calls are involved.

**Held-out validation.** A learnable group is split by prompt. Fixes are
written from one side and judged on the other, so a fix cannot win by
memorising the cases it came from. Groups with fewer than three distinct
prompts cannot be split; a fix validated that way is recorded as `in_sample`
and reported as weak evidence everywhere it appears. Pass
`require_holdout=True` to refuse those entirely.

**The ladder.** Cheapest fix first; the next rung is only tried if the
previous one loses.

1. **Template rule**: an instruction from the domain pack or from the
   feedback itself (required terms, requested format).
2. **Model-written rule**: with a `proposer` model, it reads the failures and
   writes one general instruction.
3. **Few-shot example**: a reference answer from the group. Replay never shows
   an example to the case it came from.
4. **Python tool**: with `enable_tools=True`, the proposer writes a small
   function that computes what the model keeps getting wrong. If the first
   draft does not help on the training cases, the proposer is shown what it
   returned and gets to fix it.

**The gate.** A fix is promoted only if it raises the held-out score by more
than `min_delta`, does not lower the score on other failures of the same
route, and does not lower the score on answers people approved. Everything
else is recorded as rejected, with the reason.

**Replay matches production.** Replay builds the prompt the same way live
answering does, including recall of similar past corrections (never the
case's own). A fix is measured on what it adds to the system as deployed.

**Audit.** `ai.run_audit()` switches each live patch off, replays, and rolls
back the ones whose removal costs nothing. The prompt does not grow forever,
and a patch made redundant by a later one or by a fine-tune is removed.

**Rollback.** `ai.rollback(patch_id)` undoes any rule, example, or tool. The
policy version counts every promotion and rollback, and each interaction
records the version that produced it.

## Measured results

The built-in simulator exists so tests and the demo are deterministic. Its
numbers show the plumbing working and nothing else. These are from a real
model, `phi4-mini` (3.8B) through Ollama on a laptop RTX 4050, using
`erratica benchmark`:

**Support replies** (`erratica benchmark --ollama phi4-mini:3.8b`). Eight
training prompts were asked; all eight answers failed the dataset's criteria
and received feedback. One template rule was promoted on 3 held-out cases
(0.65 -> 0.74), and a deliberately bad control rule was rejected.

| | before | after |
|---|---|---|
| 4 unseen test prompts (no feedback ever given on them) | 0.48 | 0.81 |
| 8 recorded failures, learned patches off -> on | 0.60 | 0.84 |

**Arithmetic** (`erratica benchmark --dataset arithmetic --ollama phi4-mini:3.8b
--proposer qwen3.5:4b --no-think --tools`). The model was asked to total four
to six segment revenues and got all eight training prompts wrong. Nothing was
promoted, and the 4 unseen test prompts stayed at 0.00. The proposer (another
4B model) wrote a rule, which did not help, and a tool, which was screened,
sandboxed, and used verbatim by the answering model, but summed the wrong
numbers because its parser split `90,381` at the comma. Three drafts did not
fix it. In an earlier run a model-written rule did raise the held-out score
(0.00 -> 0.33) and was still rejected, because it broke the one answer that
had been approved.

That is a negative result and it is the loop working as designed: a wrong fix
was proposed and none was installed. It also says the tool rung needs a
proposer that can write correct code, which a 4B model on this task was not.

Read these for what they are: one small model, a dozen prompts per dataset,
scored by keyword, format, and reference-similarity checks rather than by
people. They show the loop doing its job on a real model. They are not a
benchmark of how much any model improves in general. Run the command on your
own model and data to get numbers that mean something for you.

## Bring your own domain

The loop is domain-agnostic. A domain pack tells it how to recognise a kind
of task, which rule to try first for a known issue, and how to check an
answer beyond comparing it with a reference.

```python
from erratica import Domain, Erratica

def cites_sources(case, response):
    return 1.0 if "[source]" in response else 0.0

research = Domain(
    name="research",
    keywords=("paper", "study", "citation"),
    rule_templates={"missing_citation": "Cite every claim with a [source] marker."},
    issue_checks={"missing_citation": cites_sources},
    min_delta=0.1,                # stricter promotion gate for this route
)

ai = Erratica(model=my_model, domains=[research])
ai.generate("Summarise the study on sleep.")           # routed by keywords
ai.generate("Summarise this.", route="research")       # or say it yourself
```

Rules, examples, and tools are scoped to the route they were learned on. A
support rule never appears in a research prompt.

Support and finance packs are built in as examples.

## Scoring answers

Each signal available for a case contributes a score from 0 to 1: required
terms, requested format, similarity to a reference answer, the domain pack's
checks, and any critics you pass. The case score is their mean. A case with
no signal is excluded rather than counted as zero.

Similarity is pluggable:

```python
from erratica import Erratica, OllamaEmbeddingScorer, EmbeddingSimilarityScorer

ai = Erratica(similarity_scorer=OllamaEmbeddingScorer("nomic-embed-text"))   # via Ollama
ai = Erratica(similarity_scorer=EmbeddingSimilarityScorer())                 # sentence-transformers
```

The default compares wording, not meaning, so it cannot tell that two
differently phrased answers are equivalent. Use an embedding scorer, or a
critic, when paraphrases matter.

## Self-review

A critic model can grade answers nobody rated and try to write better ones:

```python
from erratica import ModelCritic

critic = ModelCritic(OllamaModel("a-larger-model"))
report = ai.self_review(critic)        # grades unrated answers, records what it finds
ai.run_learning_cycle()                # critic findings go through the same gate
```

What a critic writes is a hypothesis, and Erratica treats it as one. Critic
feedback is stored with `source='critic'`. By default it is not recalled at
answer time, not exported for fine-tuning, and counts for half in triage. It
can drive rule and example proposals, and those still have to pass replay,
including the guard on answers real people approved. Each patch records what
fraction of its evidence came from people.

A critic only helps if it is meaningfully stronger than the model it grades.

## Learned tools

Tools are off unless you pass `enable_tools=True`, because they execute
generated code. A tool is one function, `run(prompt: str) -> str`, whose
output is handed to the model as a verified note.

Two layers contain it. Static screening allows imports only from a short list
of pure-computation modules and rejects dunder names, private attributes,
file access, and dynamic execution. Then the tool runs in a separate
interpreter with restricted builtins, a timeout, and an output cap.

This is defence in depth for code you are going to read, not a security
boundary against a determined attacker. Every tool is stored in the audit
trail with its source. Read it before you rely on it.

## From mistakes to weights

```python
from erratica import CommandTrainer
from erratica.engine.llama_cpp_adapter import LlamaCppModel

trainer = CommandTrainer(
    ["python", "train_lora.py", "--data", "{dataset}", "--out", "{output}"],
    output_path="models/tuned.gguf",
    loader=lambda path: LlamaCppModel(model_path=str(path)),
)
report = ai.run_weight_cycle(trainer)

print(report.accepted, report.reason)
print(report.score_before, report.score_after)              # on held-out failures
print(report.train_score_before, report.train_score_after)  # on its own training prompts
```

A fine-tune is the easiest change to fool yourself about: train on the
corrections, test on the same prompts, and a model that only memorised them
looks perfect. So some failures are kept out of the training file and the
candidate is judged on those. It must also not get worse on approved answers.
With too few distinct failures to hold any out, the cycle refuses to run.

Erratica does not ship a GPU training stack. `CommandTrainer` runs whichever
one you use (an unsloth or PEFT script, llama.cpp tooling, a remote job). You
supply that script.

`ai.consolidate(critic=..., trainer=...)` runs self-review, a learning cycle,
an audit, and a weight cycle in one pass. Schedule `erratica consolidate` as
a nightly task to have the model tidy up what it learned while idle.

## Command line

```text
erratica demo         walk through the loop
erratica benchmark    measure the loop (--dataset support | arithmetic | file.json)
erratica cycle        run one learning cycle on a database
erratica audit        measure each live patch, roll back dead weight
erratica consolidate  cycle + audit in one pass
erratica rollback N   undo patch N
erratica dashboard    what a database has learned
erratica modelcard    a model card generated from the audit trail
erratica export       corrections as a chat-format JSONL dataset
```

Model options for the commands that replay: `--ollama NAME`, `--gguf PATH`,
`--proposer NAME` (an Ollama model that writes rules and tools), `--tools`,
`--no-think` (for reasoning models).

## Model cards

`erratica modelcard --db data/erratica.db --name my-model` writes a card from
the audit trail: what was promoted and how it was validated (held-out or
in-sample), what was rejected and why, what was rolled back, how much of the
feedback came from people, and every weight cycle with its held-out and
training scores side by side. Anyone holding the database can check it.

## What this is not

- It does not make a small model generally smarter. It fixes recurring,
  describable failures in the tasks the model is actually used for.
- The default scoring is heuristic. A patch that wins on keyword and format
  checks is not guaranteed to be better by human judgement. Add checks,
  an embedding scorer, or a critic that fit your task.
- Held-out sets are small when feedback is sparse. Three held-out cases is
  evidence, not proof. The audit exists partly to catch patches that looked
  good early and stop helping.
- Keyword routing is a convenience. Pass `route=` when you know the task.
- A single SQLite connection is used; share one `Erratica` per process.

## Development

```bash
python -m unittest        # 73 tests, no dependencies, about 20 seconds
python -m erratica demo
```

## License

MIT. See [LICENSE](LICENSE).
