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

## Certified mode

The gate described above is a heuristic: a mean has to rise by a fixed
amount on a handful of held-out cases. Each decision records a p-value so
you can see how little three cases prove (never below 0.125), and over
hundreds of patches a heuristic gate will let regressions through by chance.

Certified mode adds a second gate with a guarantee:

> With probability at least 1 - delta, over an unlimited sequence of
> candidate patches, every patch that is ever installed (a) leaves the true
> mean score of each protected behaviour less than `margin` below the
> starting system's, and (b) truly raises the mean score of its task over
> the system it replaced.

The exact statement, the assumptions, and the proof are in
[docs/THEORY.md](docs/THEORY.md).

```python
from erratica import Behaviour, Certifier, EGate, Erratica, GeneratedStream

tickets = GeneratedStream(make_ticket)          # fresh prompts on demand, each used once

certifier = Certifier(
    gate=EGate(["brevity", "grounded"], delta=0.05, margin=0.05),
    protected=[
        Behaviour("brevity", tickets, is_brief, route="support"),
        Behaviour("grounded", tickets, is_grounded, route="support"),
    ],
    tasks={"support": Behaviour("quality", tickets, quality_score, route="support")},
)
ai = Erratica(model=my_model, enable_recall=False, certifier=certifier)
```

The usual checks now act as a cheap screen. A patch that passes them is
tested on fresh samples by an e-value gate (`erratica/learning/egate.py`),
and is installed only with a certificate, which is stored in the audit
trail. How it works, briefly:

- **One test.** Paired score differences between the candidate and the
  original system feed a betting e-process. Under "the candidate is worse
  by the margin" its wealth is a supermartingale, so the chance it ever
  reaches `1/alpha` is at most `alpha`, whenever you stop.
- **The bet** hedges a mixture over a grid of bets with a
  follow-the-leader bettor. That is always valid and never more than
  `log(2K)` behind the best fixed bet on the grid.
- **Many behaviours** need no correction: a patch must pass all of them, so
  a bad patch gets through only if the test of the behaviour it harms errs.
- **Many candidates** each get a share of delta from a series that sums to
  it, decaying like `1/(k log^2 k)`, so the evidence needed grows like
  `log k`.
- **Recycled budget.** A promoted candidate hands its level to the next one
  (online fallback), which is free under arbitrary dependence.
- **Early stop.** A mirror-image test collects evidence that the candidate
  is worse than its reference at all, and ends the test when it is strong.
  That cannot cause a wrong promotion, and a candidate that is truly no
  worse is lost this way with probability at most 2%.
- **Locality.** A patch scoped to one route cannot change answers on
  another, so only behaviours on its route are re-tested.
- **Shared samples.** Behaviours on the same stream score the same
  answers, so one model call serves all of them.

**Reusing samples.** `EGate(..., reuse=True)` lets every candidate since the
last promotion be tested on the same batch of samples, with the baseline's
answers computed once for all of them. The guarantee is unchanged
(Theorem 7 in the theory note) provided the patch writer learns nothing
from a rejection except that it happened, so in this mode the details of a
rejected test are sealed until the batch is retired by a promotion or by
`ai.reveal_certificates()`.

In the simulation (8 sequences of 3,000 candidates, 90% regressions), per
installed good patch:

| setting | good patches installed | fresh evaluation items | model calls |
|---|---|---|---|
| no reuse, early stop at 2% | 98.4% | 3,351 | 10,054 |
| reuse, early stop at 2% | 98.4% | 989 | 5,379 |
| reuse, early stop at 0.5% | 99.7% | 1,008 | 6,048 |

No regression was installed in any of the three. Model calls assume one
answer from the candidate and two reference answers per item.

**Certificates you can check.** Each certificate stores the raw paired
differences of every test. `ai.verify_certificates()` replays them all and
checks that error budget was only ever passed on from a promoted candidate
to the next; `verify_certificate` does one from its JSON alone.

**After certification.** A certificate describes the system as it was. If a
model file is swapped or a scorer changes, it no longer applies.
`ai.watch()` compares the deployed system with the starting one on fresh
samples and raises an alarm on strong evidence of harm beyond the margin,
with a false-alarm probability of at most 1% over the system's whole life,
however often you look.

**What it costs.** `expected_samples(alpha, margin, gains=..., losses=...)`
estimates a test's sample cost before you run it. For a behaviour a patch
leaves untouched the cost is about `log(1/alpha) / margin`, and the docs
prove no valid test of bounded scores can beat
`log(1/alpha) / log(1/(1 - margin))`; this implementation is within about
12% of that rate.

What it assumes: scores are in [0, 1]; each evaluation sample is used once
and is independent of the patch it judges (`GeneratedStream` and
`PoolStream` enforce single use; keeping the items away from whatever
writes patches is up to you); and the scorers measure what you care about.
The margin cannot be zero, because a patch that changes nothing gives no
evidence of being better. Experience recall must be off, since recalled
corrections change answers without passing any gate; weight cycles are
refused; and only the newest patch of a route can be rolled back, because
removing one from underneath another would leave a system that was never
certified.

**Simulation.** `experiments/egate_stress.py` runs sequences of 5,000
candidates, 90% of them regressions on one protected behaviour that still
look like improvements on the task, half of those exactly on the margin
(delta 0.05, margin 0.05, 20 sequences):

| gate | regressions installed | good patches installed | samples per installed good patch, all costs included |
|---|---|---|---|
| heuristic (3 held-out cases) | 18,410 of 90,070 (20.4%) | 26% | a handful |
| e-value gate, first version | 1 of 90,070 | 57% | not measured |
| e-value gate, no early stop | 0 of 22,479 (5 sequences) | 100% | about 18,800 |
| e-value gate, current | 1 of 90,070 | 98.3% | about 3,400 |

The one regression the e-value gate installed was the first candidate of
one sequence, where the budget is largest: 1 sequence in 20, against a
guaranteed limit of 5%. A certified good patch needs about 670 to 1,030
samples itself; the rest of the all-in cost is what rejected candidates
consume. Early stopping cuts that from 2,000 samples per rejected
regression to about 260, at the price of 1.7% of good patches (the bound
is 2%). The last column is what the guarantee costs.

**On a real model.** `experiments/certified_run.py` with `phi4-mini`
(3.8B) on support tickets, delta 0.10, margin 0.10. Three candidate rules
were tried; two were written to look helpful while doing damage. Each
system was then measured on 60 fresh tickets the gate never saw.

| candidate | heuristic screen | e-value gate | measured afterwards (change vs untouched model) |
|---|---|---|---|
| verbose replies | passed | rejected after 10 tickets | brevity -0.90, quality +0.22 |
| invented reference number | passed | rejected after 15 tickets | groundedness -0.70, quality +0.34 |
| rule from the learning cycle | passed | certified after 61 tickets | quality +0.30, placeholders +0.87, nothing worse |

The heuristic gate would have installed all three. The e-value gate
installed only the one the independent measurement found harmless, using
172 model calls in total, and all three certificates replay from their
stored data. This is one model, one task, three candidates: an example of
the gate working as designed, not a measurement of its error rate.

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
- Certified mode covers rules, examples, and tools. Fine-tunes are not
  certified, and its guarantee is about your scorers on your evaluation
  stream, not about the world.

## Development

```bash
python -m unittest        # 141 tests, no dependencies, under a minute
python -m erratica demo
```

## License

MIT. See [LICENSE](LICENSE).
