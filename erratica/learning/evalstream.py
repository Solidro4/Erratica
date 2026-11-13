"""Evaluation streams: the fresh samples the e-value gate is certified on.

The gate's guarantee needs one thing from its data: a sample used to test a
candidate must be independent of that candidate. The simple way to get that,
with no assumptions about what the system saw before, is to use every sample
exactly once. A sample that has never been shown to anything cannot have
shaped the patch it is about to judge, and what earlier tests revealed about
earlier samples does not matter.

So a stream is a budget, not a benchmark you rerun. Two kinds are provided:

- ``PoolStream``: a finite set of items, consumed without replacement. When
  it runs out, certification stops; it never silently recycles.
- ``GeneratedStream``: items produced on demand by a function of a seeded
  random generator (procedural tasks with checkable answers). It never runs
  out, which is what an open-ended loop needs.

Both expose ``used`` so the position can be saved and restored. Keep the
items themselves away from whatever writes patches: nothing here can enforce
that, it is a property of how you deploy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from erratica.learning.egate import CONTINUE, PROMOTE, REJECT, BettingEProcess, EGate


class StreamExhausted(Exception):
    pass


class PoolStream:
    """A finite pool, drawn in a fixed shuffled order, each item once."""

    def __init__(self, items: Sequence[Any], *, seed: int = 0, used: int = 0) -> None:
        self._items = list(items)
        random.Random(seed).shuffle(self._items)
        self.used = used

    @property
    def remaining(self) -> int:
        return len(self._items) - self.used

    def draw(self) -> Any:
        if self.used >= len(self._items):
            raise StreamExhausted(f"all {len(self._items)} items have been used")
        item = self._items[self.used]
        self.used += 1
        return item


class GeneratedStream:
    """An unlimited stream: item i is ``make(Random(seed, i))``."""

    def __init__(self, make: Callable[[random.Random], Any], *, seed: int = 0, used: int = 0) -> None:
        self._make = make
        self.seed = seed
        self.used = used

    remaining = None  # unlimited

    def draw(self) -> Any:
        item = self._make(random.Random(f"{self.seed}:{self.used}"))
        self.used += 1
        return item


class BatchCache:
    """The evaluation batch currently being reused.

    Holds, per stream, the items drawn so far and the baseline's and the
    incumbent's answers to them. Every candidate tested on the batch reads
    the items in the same order from the start, drawing new ones only when
    it needs more than any earlier candidate did.
    """

    def __init__(self) -> None:
        self._items: dict = {}
        self._answers: dict = {}

    def item(self, stream: Any, index: int):
        """The index-th item of the batch for this stream, and whether it
        had to be freshly drawn."""
        items = self._items.setdefault(id(stream), [])
        fresh = False
        while len(items) <= index:
            items.append(stream.draw())
            fresh = True
        return items[index], fresh

    def answer(self, role: str, stream: Any, index: int, compute: Callable[[], str]) -> str:
        key = (role, id(stream), index)
        if key not in self._answers:
            self._answers[key] = compute()
        return self._answers[key]

    def clear(self) -> None:
        self._items.clear()
        self._answers.clear()

    def __len__(self) -> int:
        return sum(len(items) for items in self._items.values())


@dataclass
class Behaviour:
    """One thing to measure: where its samples come from and how to score them."""

    name: str
    stream: Any
    # (item, response) -> score in [0, 1]
    score: Callable[[Any, str], float]
    # item -> the prompt to send; defaults to the item itself
    prompt: Optional[Callable[[Any], str]] = None
    # The route these prompts belong to, when known. A patch scoped to a
    # different route cannot change the answers, so it is not re-tested.
    route: Optional[str] = None

    def ask(self, model: Callable[[str, Optional[str]], str], item: Any) -> float:
        text = self.prompt(item) if self.prompt else item
        return float(self.score(item, model(text, self.route)))


def certify(
    gate: EGate,
    *,
    candidate: Callable[[str, Optional[str]], str],
    baseline: Callable[[str, Optional[str]], str],
    protected: Sequence[Behaviour],
    task: Optional[Behaviour] = None,
    incumbent: Optional[Callable[[str, Optional[str]], str]] = None,
    cache: Optional[BatchCache] = None,
) -> dict:
    """Run one candidate through the gate on fresh samples.

    With ``cache`` (and a gate created with ``reuse=True``) the samples are
    those of the current evaluation batch, shared with the other
    candidates tested since the last promotion, and the baseline's and
    incumbent's answers are computed once per item for all of them. The
    caller must clear the cache when the gate moves to a new batch, and
    must keep details of rejected tests away from whatever writes patches
    until then.

    ``candidate``, ``baseline`` (the original system, the anchor of the
    guarantee) and ``incumbent`` (what is deployed now; defaults to the
    baseline) are functions ``(prompt, route) -> answer``; ``route`` is the
    behaviour's route or None.

    ``protected`` may be a subset of the gate's behaviours when the
    candidate cannot affect the rest; only the ones passed are tested.

    Behaviours that share a stream object share its samples: one item is
    drawn, each system answers it once, and every behaviour on that stream
    scores the same answers. That is valid (the tests may depend on each
    other arbitrarily) and it divides the number of model calls by the
    number of behaviours sharing the stream. Such behaviours must agree on
    how an item becomes a prompt. Sampling for a behaviour stops as soon as
    its test is settled.

    Returns the gate's certificate with the decision, plus what was spent.
    A stream running dry is a rejection, not an error: without fresh
    samples nothing can be certified.
    """
    names = [behaviour.name for behaviour in protected]
    incumbent = incumbent or baseline
    test = gate.begin(behaviours=names, require_improvement=task is not None)

    # Group behaviours by the stream they read from.
    groups: list = []
    for behaviour in list(protected) + ([task] if task is not None else []):
        for group in groups:
            if group["stream"] is behaviour.stream:
                break
        else:
            group = {"stream": behaviour.stream, "protected": [], "task": None}
            groups.append(group)
        if behaviour is task:
            group["task"] = behaviour
        else:
            group["protected"].append(behaviour)

    consumed = {behaviour.name: 0 for behaviour in protected}
    if task is not None:
        consumed[task.name] = 0
    spent = {"items": 0, "fresh_items": 0, "model_calls": 0}
    note = ""
    if cache is not None and not gate.reuse:
        raise ValueError("a batch cache needs a gate created with reuse=True")

    def call(model, text, route):
        spent["model_calls"] += 1
        return model(text, route)

    def reference(role, model, stream, index, text, route):
        if cache is None:
            return call(model, text, route)
        return cache.answer(role, stream, index, lambda: call(model, text, route))

    try:
        while test.decision == CONTINUE:
            for group in groups:
                active = [b for b in group["protected"] if not test.settled(b.name)]
                task_active = group["task"] is not None and not test.settled(None)
                if not active and not task_active:
                    continue
                stream = group["stream"]
                index = group.setdefault("index", 0)
                group["index"] = index + 1
                if cache is None:
                    item, fresh = stream.draw(), True
                else:
                    item, fresh = cache.item(stream, index)
                spent["items"] += 1
                spent["fresh_items"] += int(fresh)
                lead = active[0] if active else group["task"]
                text = lead.prompt(item) if lead.prompt else item
                candidate_answer = call(candidate, text, lead.route)
                baseline_answer = (
                    reference("baseline", baseline, stream, index, text, lead.route) if active else None
                )
                for behaviour in active:
                    consumed[behaviour.name] += 1
                    test.add_protected(
                        behaviour.name,
                        float(behaviour.score(item, candidate_answer)),
                        float(behaviour.score(item, baseline_answer)),
                    )
                if task_active:
                    if incumbent is baseline and baseline_answer is not None:
                        incumbent_answer = baseline_answer
                    elif incumbent is baseline:
                        incumbent_answer = reference("baseline", baseline, stream, index, text, lead.route)
                    else:
                        incumbent_answer = reference("incumbent", incumbent, stream, index, text, lead.route)
                    consumed[group["task"].name] += 1
                    test.add_task(
                        float(group["task"].score(item, candidate_answer)),
                        float(group["task"].score(item, incumbent_answer)),
                    )
                if test.decision != CONTINUE:
                    break
    except StreamExhausted as exc:
        note = f"evaluation stream ran out: {exc}"

    certificate = test.certificate()
    if note and certificate["decision"] != PROMOTE:
        certificate["decision"] = REJECT
        certificate["note"] = note
    certificate["samples_consumed"] = consumed
    certificate["items_drawn"] = spent["items"]
    certificate["fresh_items"] = spent["fresh_items"]
    certificate["model_calls"] = spent["model_calls"]
    certificate["delta"] = gate.delta
    certificate["not_retested"] = sorted(set(gate.behaviours) - set(names))
    return certificate


class DriftMonitor:
    """An alarm for the day the certificate stops being true.

    A certificate is about the system and the evaluation stream as they
    were when it was issued. Models get swapped, scorers change, the world
    moves. The monitor keeps comparing the deployed system with the
    baseline on fresh samples and raises an alarm when there is strong
    evidence that a protected behaviour is now worse by the margin or more.

    Each behaviour has one e-process that runs forever. If the deployed
    system has always been within the margin, the chance that a given
    behaviour ever raises an alarm is at most ``alarm`` divided by the
    number of behaviours, however long it runs and however often you look.
    """

    def __init__(self, behaviours: Sequence[str], margin: float, *, alarm: float = 0.01) -> None:
        self.margin = margin
        self.alarm = alarm
        self.level = alarm / max(1, len(behaviours))
        self._differences: dict = {name: [] for name in behaviours}
        self._processes: dict = {name: self._new_process() for name in behaviours}

    def _new_process(self) -> BettingEProcess:
        # y = (baseline - deployed) - margin; H0 "within the margin" is E[y] <= 0.
        return BettingEProcess(lower=-1.0 - self.margin, upper=1.0 - self.margin)

    def add(self, behaviour: str, deployed: float, baseline: float) -> None:
        difference = float(deployed) - float(baseline)
        self._differences[behaviour].append(difference)
        self._processes[behaviour].update(-difference - self.margin)

    def alarms(self) -> list:
        return sorted(
            name for name, process in self._processes.items() if process.crossed(self.level)
        )

    def report(self) -> dict:
        return {
            name: {
                "samples": process.n,
                "mean_change": (
                    sum(self._differences[name]) / process.n if process.n else None
                ),
                "log_evidence_of_harm": process.max_log_wealth,
                "alarm": process.crossed(self.level),
            }
            for name, process in self._processes.items()
        }

    def state(self) -> dict:
        return {name: list(values) for name, values in self._differences.items()}

    def restore(self, state: Optional[dict]) -> None:
        for name, differences in (state or {}).items():
            if name not in self._processes:
                continue
            self._differences[name] = []
            self._processes[name] = self._new_process()
            for difference in differences:
                self._differences[name].append(float(difference))
                self._processes[name].update(-float(difference) - self.margin)


@dataclass
class Certifier:
    """Everything certified mode needs: the gate, and where its samples come from.

    ``protected`` are the behaviours that must never get worse than the
    starting system by the gate's margin. ``tasks`` maps a route to the
    behaviour a patch on that route is supposed to improve; a route with
    no entry is certified for non-regression only.
    """

    gate: EGate
    protected: Sequence[Behaviour] = ()
    tasks: Optional[dict] = None
    # Chance of a false drift alarm over the whole lifetime of the system.
    alarm: float = 0.01

    def __post_init__(self) -> None:
        names = [behaviour.name for behaviour in self.protected]
        if sorted(names) != sorted(self.gate.behaviours):
            raise ValueError(
                f"gate protects {sorted(self.gate.behaviours)} but streams were given for {sorted(names)}"
            )
        self.tasks = dict(self.tasks or {})
        for route, behaviour in self.tasks.items():
            if behaviour.route is None:
                behaviour.route = route
            elif behaviour.route != route:
                raise ValueError(
                    f"task '{behaviour.name}' is registered for route '{route}' but says '{behaviour.route}'"
                )
        # Behaviours that share a stream share its samples and answers, so
        # they have to agree on what route those samples belong to.
        everything = list(self.protected) + list(self.tasks.values())
        for first in everything:
            for second in everything:
                if first.stream is second.stream and first.route != second.route:
                    raise ValueError(
                        f"'{first.name}' and '{second.name}' share a stream but have different routes"
                    )
        self.monitor = DriftMonitor(names, self.gate.margin, alarm=self.alarm)
        # The evaluation batch being reused (only used when gate.reuse).
        self.cache = BatchCache()

    def affected_by(self, route: Optional[str]) -> list:
        """Protected behaviours a patch scoped to ``route`` could change."""
        return [
            behaviour
            for behaviour in self.protected
            if route is None or behaviour.route is None or behaviour.route == route
        ]

    def _streams(self) -> dict:
        streams = {f"protected:{behaviour.name}": behaviour.stream for behaviour in self.protected}
        for route, behaviour in self.tasks.items():
            streams[f"task:{route}"] = behaviour.stream
        return streams

    def state(self) -> dict:
        """Gate budget and stream positions, to be saved between sessions."""
        return {
            "gate": self.gate.state(),
            "streams": {name: stream.used for name, stream in self._streams().items()},
            "monitor": self.monitor.state(),
        }

    def restore(self, state: Optional[dict]) -> None:
        if not state:
            return
        gate_state = state.get("gate", {})
        self.gate.tested = int(gate_state.get("tested", self.gate.tested))
        self.gate._carry = float(gate_state.get("carry", 0.0))
        self.gate._last = None
        self.gate.batch = int(gate_state.get("batch", self.gate.batch))
        self.gate.position = int(gate_state.get("position", 0))
        # The cached samples of a batch do not survive a restart, so a
        # batch that was in use is retired and the next candidate starts a
        # new one. Costs a little budget; never costs validity.
        self.cache.clear()
        self.gate.retire_batch()
        # Behaviours may share one stream object; it must end up at the
        # furthest position any of them recorded.
        positions: dict = {}
        for name, stream in self._streams().items():
            if name in state.get("streams", {}):
                key = id(stream)
                positions[key] = max(positions.get(key, 0), int(state["streams"][name]))
        for stream in self._streams().values():
            if id(stream) in positions:
                stream.used = positions[id(stream)]
        self.monitor.restore(state.get("monitor"))
