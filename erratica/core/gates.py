"""Gates: who decides whether a change may be installed.

A gate is any callable ``gate(trial) -> Verdict``. It may also return a bool,
or ``(bool, reason)``. The trial hands it three systems (the candidate, the
incumbent that is deployed now, and the baseline the loop started from) and
the recorded cases, and scores them on demand so a gate only pays for the
comparisons it actually asks for.

Two gates are provided, and ``all_of`` chains gates so a cheap one screens
candidates before an expensive one:

- ``ReplayGate`` replays recorded cases. Fast, needs no extra data, and is a
  heuristic: a handful of cases is evidence, not proof.
- ``CertifiedGate`` tests on fresh samples with the e-value gate
  (``erratica.learning.egate``), which bounds the chance of ever installing
  a harmful change.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from erratica.core.types import HOLDOUT, Case, Change, System, Verdict
from erratica.learning.promoter import paired_p_value, why_not


def noise_p_value(
    before: Sequence[Sequence[float]],
    after: Sequence[Sequence[float]],
    *,
    exact_up_to: int = 50000,
    draws: int = 10000,
) -> Optional[float]:
    """Chance of a gain this large if the change did nothing, measured
    against the systems' own run-to-run noise.

    Each case was answered several times by each system. If the change
    does nothing, a case's scores are the same kind of draw whichever
    system produced them, so they can be relabelled freely within the
    case. The observed total gain is compared with the gains under every
    relabelling (enumerated when there are few, sampled with a fixed seed
    otherwise).

    A system that always answers the same way has no noise, so two cases
    that both move are already strong evidence (p = 1/36 with two repeats).
    A model that words things differently on every run has to beat its own
    scatter. This says the gain is real *on these cases*; how far it
    carries to new ones is what holding cases out is for.
    """
    if not before or any(len(b) < 2 or len(a) < 2 for b, a in zip(before, after)):
        return None
    observed = 0.0
    per_case: List[List[float]] = []
    arrangements = 1
    for b, a in zip(before, after):
        pooled = list(b) + list(a)
        total, size = sum(pooled), len(a)
        observed += sum(a) / len(a) - sum(b) / len(b)
        gains = []
        for chosen in itertools.combinations(range(len(pooled)), size):
            picked = sum(pooled[i] for i in chosen)
            gains.append(picked / size - (total - picked) / (len(pooled) - size))
        per_case.append(gains)
        arrangements *= len(gains)
    if arrangements <= exact_up_to:
        hits = sum(sum(combo) >= observed - 1e-12 for combo in itertools.product(*per_case))
        return hits / arrangements
    rng = random.Random(0)
    hits = sum(sum(rng.choice(gains) for gains in per_case) >= observed - 1e-12 for _ in range(draws))
    return (hits + 1) / (draws + 1)


@dataclass
class Paired:
    """Scores of the same cases under two systems. ``before`` and ``after``
    are per-case means; the ``_samples`` lists hold every repeat."""

    before: List[float] = field(default_factory=list)
    after: List[float] = field(default_factory=list)
    before_samples: List[Sequence[float]] = field(default_factory=list)
    after_samples: List[Sequence[float]] = field(default_factory=list)

    @property
    def p_value(self) -> Optional[float]:
        """With repeats, the gain against run-to-run noise; with one answer
        per case, a sign-flip test across cases (which cannot go below
        ``1 / 2**n`` for n cases)."""
        noise = noise_p_value(self.before_samples, self.after_samples)
        return noise if noise is not None else paired_p_value(self.deltas)

    @property
    def n(self) -> int:
        return len(self.before)

    @property
    def mean_before(self) -> float:
        return sum(self.before) / self.n if self.n else 0.0

    @property
    def mean_after(self) -> float:
        return sum(self.after) / self.n if self.n else 0.0

    @property
    def deltas(self) -> List[float]:
        return [after - before for before, after in zip(self.before, self.after)]

    @property
    def gain(self) -> float:
        return self.mean_after - self.mean_before


class Trial:
    """One candidate up for judgement.

    ``candidate`` and ``incumbent`` are systems; ``baseline`` is given as a
    function that returns the baseline system, so it is only built when a
    gate needs it. ``target`` are the failures the candidate is meant to fix (the held-out
    ones when there were enough to hold any out). ``approved`` are answers
    people accepted and ``others`` are failures of other issues; a candidate
    should not make either worse.
    """

    def __init__(
        self,
        *,
        candidate: System,
        incumbent: System,
        baseline: Callable[[], System],
        target: Sequence[Case],
        approved: Sequence[Case],
        others: Sequence[Case],
        validation: str,
        scorer: Callable[[str, Sequence[Case]], List[Optional[Sequence[float]]]],
        change: Optional[Change] = None,
        issue: Optional[str] = None,
        name: Optional[str] = None,
        incumbent_is_baseline: bool = False,
    ) -> None:
        self.candidate, self.incumbent = candidate, incumbent
        self._baseline = baseline
        self.target, self.approved, self.others = list(target), list(approved), list(others)
        self.validation = validation
        self.change, self.issue = change, issue
        self.name = name or (change.name if change is not None else "candidate")
        self.incumbent_is_baseline = incumbent_is_baseline
        self._scorer = scorer

    @property
    def baseline(self) -> System:
        """The system the loop started from. Built only when a gate asks."""
        return self._baseline()

    def samples(self, role: str, cases: Sequence[Case]) -> List[Optional[Sequence[float]]]:
        """Every repeat's score for ``cases`` under ``"candidate"``,
        ``"incumbent"`` or ``"baseline"`` (None where no check applies)."""
        return self._scorer(role, cases)

    def scores(self, role: str, cases: Sequence[Case]) -> List[Optional[float]]:
        """Mean score per case under a role."""
        return [None if s is None else sum(s) / len(s) for s in self.samples(role, cases)]

    def compare(self, cases: Sequence[Case], *, against: str = "incumbent") -> Paired:
        """The candidate against another role on the same cases. Cases that
        no check applies to under either system are left out."""
        paired = Paired()
        if not cases:
            return paired
        for before, after in zip(self.samples(against, cases), self.samples("candidate", cases)):
            if before is None or after is None:
                continue
            paired.before.append(sum(before) / len(before))
            paired.after.append(sum(after) / len(after))
            paired.before_samples.append(before)
            paired.after_samples.append(after)
        return paired

    def verdict(self, accepted: bool, reason: str, **fields: Any) -> Verdict:
        return Verdict(
            accepted=accepted,
            reason=reason,
            change=self.name,
            issue=self.issue,
            validation=self.validation,
            **fields,
        )


class ReplayGate:
    """Accept a change that fixes recorded failures without breaking the rest.

    A candidate passes when, on the target failures, its mean score beats
    the incumbent's by more than ``min_gain`` and that gain is unlikely
    under "the change does nothing" (p-value at most ``max_p``), and on
    approved answers and on other failures its mean score does not fall by
    more than ``max_loss``.

    The p-value is measured against the system's own run-to-run noise when
    each case was answered more than once (the loop's ``repeats``, two by
    default). A model that rewords its answer on every run then has to
    beat its own scatter, and a system that always answers the same way
    passes on two cases. With ``repeats=1`` it falls back to a test across
    cases, which needs at least three cases to get under the default.
    ``max_p=None`` switches the test off. ``require_holdout`` refuses
    anything that could only be judged on the failures it was written from.

    With no recorded failures to improve on, only the guards are checked.
    That is the "did my new version break anything?" question.
    """

    name = "replay"

    def __init__(
        self,
        *,
        min_gain: float = 0.05,
        max_loss: float = 0.02,
        max_p: Optional[float] = 0.2,
        require_holdout: bool = False,
    ) -> None:
        self.min_gain, self.max_loss = min_gain, max_loss
        self.max_p, self.require_holdout = max_p, require_holdout

    def __call__(self, trial: Trial) -> Verdict:
        if self.require_holdout and trial.validation != HOLDOUT:
            return trial.verdict(False, "too few distinct failures to hold any out", gate=self.name)

        fields: Dict[str, Any] = {"gate": self.name}
        if trial.target:
            target = trial.compare(trial.target)
            if target.n == 0:
                return trial.verdict(
                    False,
                    "no check applies to the target failures; record expect=, must= or add a check",
                    **fields,
                )
            fields.update(
                before=target.mean_before,
                after=target.mean_after,
                cases=target.n,
                p_value=target.p_value,
            )
            if target.gain <= self.min_gain:
                return trial.verdict(
                    False,
                    f"gain {target.gain:+.3f} on {target.n} target cases is not above {self.min_gain}",
                    **fields,
                )
            if self.max_p is not None and fields["p_value"] is not None and fields["p_value"] > self.max_p:
                return trial.verdict(
                    False,
                    f"gain {target.gain:+.3f} on {target.n} target cases could be chance "
                    f"(p = {fields['p_value']:.3f}, limit {self.max_p}); more cases or more repeats would tell",
                    **fields,
                )

        # The guards are only replayed for a candidate worth guarding.
        worst = 0.0
        guarded = 0
        for label, cases in (("approved answers", trial.approved), ("other failures", trial.others)):
            guard = trial.compare(cases)
            guarded += guard.n
            if cases is trial.approved and guard.n < len(cases):
                # An approval no check can score protects nothing. Say so.
                fields["evidence"] = {"approved_unchecked": len(cases) - guard.n}
            if guard.n == 0:
                continue
            worst = max(worst, -guard.gain)
            if -guard.gain > self.max_loss:
                return trial.verdict(
                    False,
                    f"lowers the score on {label} ({guard.mean_before:.3f} -> {guard.mean_after:.3f}, "
                    f"{guard.n} cases)",
                    guard_loss=-guard.gain,
                    **fields,
                )
        fields["guard_loss"] = worst
        unchecked = fields.get("evidence", {}).get("approved_unchecked", 0)
        note = f"; {unchecked} approved answers have no check that applies and were not guarded" if unchecked else ""
        if not trial.target:
            if not guarded:
                # Passing a candidate nothing was measured on would be a
                # promise with nothing behind it.
                return trial.verdict(
                    False,
                    "nothing to judge on: no failure is recorded and no approved answer has a check "
                    "that applies. Record a failure, or approve an answer and say what made it good",
                    **fields,
                )
            return trial.verdict(True, "no recorded failures to improve on; the guards hold" + note, **fields)
        return trial.verdict(True, "improves the target failures and the guards hold" + note, **fields)


class CertifiedGate:
    """Install only with a certificate from the e-value gate.

    ``gate`` is an ``EGate``; ``protected`` are ``Behaviour`` objects, each
    with a stream of fresh samples and a scorer, that must never fall more
    than the gate's margin below the baseline; ``task`` is the behaviour a
    change is supposed to improve (omit it to certify "does no harm" only).

    The guarantee, its assumptions and its proof are in ``docs/THEORY.md``.
    In short: every sample is used once, the candidate is compared with the
    baseline this loop was built with, and with probability at least
    ``1 - delta`` no change that harms a protected behaviour beyond the
    margin is ever installed, however many candidates are tried.

    Put a ``ReplayGate`` in front of it with ``all_of``: budget is spent on
    every candidate that reaches this gate, so screening out the obvious
    losers first keeps later tests cheap.
    """

    name = "certified"
    certified = True

    def __init__(self, gate: Any, *, protected: Sequence[Any] = (), task: Any = None) -> None:
        from erratica.learning.evalstream import Certifier

        if gate.reuse:
            raise ValueError(
                "CertifiedGate does not support EGate(reuse=True): reuse needs rejected tests "
                "to be sealed, which only the Erratica class does"
            )
        self.certifier = Certifier(gate=gate, protected=list(protected))
        self.task = task

    @staticmethod
    def _ask(system: System) -> Callable[[Any, Optional[str]], Any]:
        def answer(item: Any, route: Optional[str]) -> Any:
            try:
                return system(item)
            except Exception:  # a crashing system gave no answer
                return ""

        return answer

    def __call__(self, trial: Trial) -> Verdict:
        from erratica.learning.evalstream import certify

        certificate = certify(
            self.certifier.gate,
            candidate=self._ask(trial.candidate),
            baseline=self._ask(trial.baseline),
            incumbent=None if trial.incumbent_is_baseline else self._ask(trial.incumbent),
            protected=list(self.certifier.protected),
            task=self.task,
        )
        accepted = certificate.get("decision") == "promote"
        reason = (
            f"certified on {certificate.get('fresh_items', 0)} fresh samples"
            if accepted
            else why_not(certificate)
        )
        return trial.verdict(accepted, reason, gate=self.name, evidence={"certificate": certificate})

    def state(self) -> dict:
        state = {"certifier": self.certifier.state()}
        if self.task is not None:
            state["task_used"] = self.task.stream.used
        return state

    def restore(self, state: Optional[dict]) -> None:
        if not state:
            return
        self.certifier.restore(state.get("certifier"))
        if self.task is not None and "task_used" in state:
            self.task.stream.used = max(self.task.stream.used, int(state["task_used"]))


class all_of:  # noqa: N801 - reads as a function at the call site
    """Chain gates. The first rejection ends the trial; the last gate's
    verdict stands, carrying the measurements of the ones before it."""

    def __init__(self, *gates: Callable[[Trial], Any]) -> None:
        if not gates:
            raise ValueError("all_of needs at least one gate")
        self.gates = gates
        self.name = " + ".join(getattr(gate, "name", getattr(gate, "__name__", "gate")) for gate in gates)
        self.certified = any(getattr(gate, "certified", False) for gate in gates)

    def __call__(self, trial: Trial) -> Verdict:
        merged: Optional[Verdict] = None
        for gate in self.gates:
            verdict = as_verdict(gate(trial), trial, gate)
            if merged is not None:
                for name in ("before", "after", "p_value", "guard_loss"):
                    if getattr(verdict, name) is None:
                        setattr(verdict, name, getattr(merged, name))
                verdict.cases = verdict.cases or merged.cases
                verdict.evidence = {**merged.evidence, **verdict.evidence}
            merged = verdict
            if not verdict.accepted:
                break
        merged.gate = self.name
        return merged

    def state(self) -> dict:
        return {
            str(index): gate.state()
            for index, gate in enumerate(self.gates)
            if hasattr(gate, "state")
        }

    def restore(self, state: Optional[dict]) -> None:
        for index, gate in enumerate(self.gates):
            if hasattr(gate, "restore") and state and str(index) in state:
                gate.restore(state[str(index)])


def as_verdict(result: Any, trial: Trial, gate: Any) -> Verdict:
    """Accept a Verdict, a bool, or ``(bool, reason)`` from a gate."""
    name = getattr(gate, "name", getattr(gate, "__name__", "gate"))
    if isinstance(result, Verdict):
        result.change = result.change or trial.name
        result.issue = result.issue or trial.issue
        result.gate = result.gate or name
        return result
    if isinstance(result, tuple) and len(result) == 2:
        return trial.verdict(bool(result[0]), str(result[1]), gate=name)
    return trial.verdict(bool(result), "accepted" if result else "rejected", gate=name)
