# The composable core

`erratica.core` is the part of Erratica that makes no assumptions about your
system. It does one job: **it controls how an AI system is allowed to
change.** Mistakes go on a record, candidate changes are judged against that
record, and only a change the gate accepts is installed. Everything else is
yours to shape.

This page is the reference. For a first look, run `examples/quickstart.py`.

## The idea in four lines

| Thing | What it is | Who owns it |
|---|---|---|
| System | any callable `input -> output` | you |
| Case | a recorded mistake or an approved answer, as plain data | the record |
| Change | a list of edits to a configuration, as plain data | the record |
| Stages | checks, gate, proposers, store: small functions or objects | you, or the built-ins |

Cases and changes are data so they can be stored, shown in a diff, replayed
and undone. Code lives only in the stages, and every stage is an argument of
`Loop`, so replacing one never means touching another.

## Two ways in

**Judge mode.** You build and deploy the system yourself. The loop keeps the
record and answers one question: is this new version safe to ship?

```python
from erratica import Loop

loop = Loop(my_agent)                         # any callable
run = loop.run("Summarise the sleep study")
loop.fail(run, issue="missing_citation", must=["[source]"])
loop.approve(other_run, must=["p-value"])

verdict = loop.judge(my_new_agent)            # nothing is installed
print(verdict.accepted, verdict.reason, verdict.before, verdict.after)
if verdict:
    loop.adopt(my_new_agent, verdict=verdict) # keep the loop in step with what you deployed
```

**Managed mode.** You give a configuration and a `build(config)` function.
Now the loop can propose changes itself, install them, audit them and roll
them back.

```python
from erratica import Loop, prompted
from erratica.core.proposers import rules_from_feedback, rule_from_model

loop = Loop(
    build=prompted(my_llm, base="You are a research assistant."),
    config={"rules": []},
    proposers=[rules_from_feedback(), rule_from_model(my_writer)],
    store="errata.db",
)
...
loop.learn()       # propose -> judge on held-out failures -> install
loop.audit()       # switch each change off; drop the ones that stopped helping
loop.rollback()    # undo the newest change
```

`build` is where your framework goes. A chain, a graph, a crew of agents, a
retrieval pipeline: build it from the configuration and return a function
that runs it.

```python
def build(config):
    agent = make_my_agent(instructions=config["rules"], model=config["model"])
    return lambda request: agent.run(request)
```

## The stages

### Checks: how an answer is scored

A check is `(case, output) -> score`: a number from 0 to 1, `True` / `False`,
or `None` when it has nothing to say about this case. A case's score is the
mean of the checks that apply. A case no check applies to is left out of
every average; it is never counted as zero.

```python
from erratica import checks

@checks.check(issues=["wrong_total"])          # only for this issue
def right_total(case, output):
    return output == case.meta["total"]

loop = Loop(my_system, checks=[checks.terms, right_total, checks.max_words(120)])
```

| Built in | Scores |
|---|---|
| `checks.terms` | the case's `must` terms present, `must_not` terms absent |
| `checks.reference(scorer)` | closeness to `case.expect`, the answer that was wanted |
| `checks.equals` | exact match with `case.expect` |
| `checks.contains(...)`, `excludes(...)`, `matches(regex)` | every answer |
| `checks.max_words(n)`, `checks.valid_json(*keys)` | every answer |
| `checks.judged(model, rubric)` | a model grades the answer (a hypothesis, not a fact) |
| `checks.stays_close_to_approved(scorer)` | for approved answers, with a scorer that compares meaning |

With no `checks=` the loop uses `terms` and `reference()`.

### Gate: who decides

A gate is `gate(trial) -> Verdict`. It may also return a bool or
`(bool, reason)`. The trial gives it the candidate, the incumbent (what is
deployed now), the baseline (what the loop started from) and the cases, and
scores them only when asked.

```python
def careful(trial):
    target = trial.compare(trial.target)         # candidate vs incumbent
    if target.mean_after < 1.0:
        return False, "does not fix every target case"
    return trial.compare(trial.approved).gain >= 0, "approved answers hold"
```

| Built in | Use it when |
|---|---|
| `ReplayGate(min_gain, max_loss, max_p, require_holdout)` | you want a fast answer from the recorded cases. The gain must stand out from run-to-run noise, but a few cases are still evidence, not proof. |
| `CertifiedGate(EGate(...), protected=[...], task=...)` | a wrong install is expensive. Tests on fresh samples and bounds the chance of ever installing a harmful change. See [THEORY.md](THEORY.md). |
| `all_of(gate, gate, ...)` | chaining. Put the cheap gate first so the expensive one only sees candidates worth its budget. |

### Proposers: where changes come from

A proposer is `proposer(context) -> changes`. It sees the issue, the current
configuration and the failures on the training side of the split, never the
ones the gate will judge on. The loop tries proposers in order and stops at
the first accepted change, so list the cheapest idea first.

```python
from erratica import Change

def wider_search(context):
    if context.issue == "missed_document":
        yield Change.set("retriever.top_k", context.config["retriever"]["top_k"] + 2)
```

| Built in | Proposes |
|---|---|
| `rules_from_feedback(templates)` | an instruction from a template or from the recorded `must` terms. No model call. |
| `rule_from_model(model)` | one general instruction a model writes after reading the failures |
| `examples_from_failures()` | a worked example, only when it can be judged on other cases |
| `try_values(path, values)` | other values for any setting: a model name, a temperature, a retrieval depth |

### Store: where the record lives

`store=None` keeps it in memory, a path opens one SQLite file with JSON rows,
and any object with `add`, `put`, `rows`, `get_state` and `set_state` works
too. With a persistent store, a new `Loop` on the same file picks up the
cases, the installed changes, the version and the gate's budget.

### Events

`loop.on("verdict", handler)` and likewise for `"run"`, `"case"`,
`"install"` and `"rollback"`. Use them to post to a channel or write a log.

## Changes

```python
Change.set("model", "large")
Change.append("rules", "Cite every claim.")
Change.remove("rules", "Be brief.")
Change.set("retriever.top_k", 5) + Change.set("retriever.rerank", True)   # one change
```

Paths are dotted. A change never edits your configuration in place; the
current configuration is always the starting one with the active changes
applied in order, which is why any one of them can be taken out later.

## What the evidence means

Every verdict carries its evidence: the scores before and after, how many
cases, a p-value, the worst drop on the guarded groups, and how it was
validated.

- **`holdout`**: the change was written from some failures and judged on
  others. `learn` does this whenever an issue has three or more distinct
  failures.
- **`in_sample`**: the author of the change could see the cases it was judged
  on. That covers small issues in `learn` and anything you write yourself and
  pass to `judge` or `install`. It is weaker evidence and is labelled so
  everywhere. `ReplayGate(require_holdout=True)` refuses it.
- **p-value**: the chance of a gain this large if the change did nothing.
  By default every case is answered twice by each system (`repeats=2`) and
  the gain is tested against the systems' own run-to-run noise: a model
  that rewords its answer on every run has to beat its own scatter, and a
  system that always answers the same way passes on two cases. `ReplayGate`
  refuses a gain with a p-value above `max_p` (0.2 by default). This says
  the gain is real on the cases it was measured on. How far it carries to
  new ones depends on how many cases there were and whether they were held
  out. With `repeats=1` the test runs across cases instead and cannot go
  below 0.125 with three of them.
- **certificate**: with `CertifiedGate`, the raw paired differences of the
  test, checkable by anyone with `verify_certificate`.

## What an approval protects

An approval protects an answer through the checks, so say what made the
answer good (`must=[...]`, `meta=...`) or use checks that apply to every
answer. The approved wording is stored but is not treated as the one right
answer.

That is a measured decision. On `phi4-mini` (3.8B), eight support requests
were answered, then answered again under four prompts, and each new answer
was compared with the first one:

| Prompt change | Wording similarity to the first answer | Shared content words |
|---|---|---|
| none (same prompt again) | 0.88 | 0.93 |
| one harmless rule ("Address the customer politely.") | 0.26 | 0.51 |
| one useful rule (numbered list with resolution and timeline) | 0.19 | 0.36 |
| one harmful rule (single short sentence, no detail) | 0.24 | 0.18 |

Wording similarity cannot tell the harmless rule from the harmful one, so a
guard built on it would refuse every change or none. If a verdict says some
approved answers "have no check that applies and were not guarded", that is
the loop telling you an approval is protecting nothing yet.

## Recipes

**A gate in CI.** Keep the record in the repository or a shared file, and
fail the build when a new version breaks it.

```python
loop = Loop(current_agent, store="errata.db")
verdict = loop.judge(agent_from_this_branch)
assert verdict.accepted, verdict.reason
```

**A system that does not answer the same way twice.** Raise the number of
answers per case and ask for stronger evidence:
`Loop(..., repeats=4, gate=ReplayGate(max_p=0.05))`. Add `workers=8` to
answer cases in parallel when the model is behind a network call.

**A system that always answers the same way, and is expensive to call.**
`Loop(..., repeats=1, gate=ReplayGate(max_p=None))` answers each case once
and skips the noise test.

**A test set kept aside.** `loop.measure(cases)` scores cases that are not
on the record and never shows them to a proposer.

**Certified installs.**

```python
from erratica import CertifiedGate, EGate, Behaviour, GeneratedStream, ReplayGate, all_of

requests = GeneratedStream(make_request)        # fresh inputs on demand, each used once
gate = all_of(
    ReplayGate(),
    CertifiedGate(
        EGate(["brevity"], delta=0.05, margin=0.05),
        protected=[Behaviour("brevity", requests, is_brief)],
        task=Behaviour("quality", requests, quality_score),
    ),
)
loop = Loop(build=build, config=config, gate=gate, store="errata.db")
```

With a certified gate, a forced install and an audit are refused and only
the newest change can be rolled back, because each of those would leave a
system that was never certified.

## Limits

- The default checks are simple. A change that wins on terms and a reference
  answer is not guaranteed to be better by human judgement. Add checks that
  express what you care about.
- A recorded input is replayed later, so with a persistent store it has to
  be plain JSON.
- `learn` works on one issue at a time and stops at the first accepted
  change. An issue is looked at again when it has new failures.
- In judge mode the loop cannot roll back or audit: you deployed the system,
  so you undo it.
- One process should own a store. Runs can be recorded from several threads;
  judging and learning should happen in one.
- The core does not yet include the tool-writing rung, experience recall or
  fine-tuning. Those are in the `Erratica` class.
