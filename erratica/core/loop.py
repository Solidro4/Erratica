"""The loop: record mistakes, try changes, install only what a gate accepts.

Two ways to use it, and you can start with the first and grow into the second.

**Judge mode**: you own the system and its deployment. Erratica keeps the
record of mistakes and approved answers and tells you whether a new version
is safe to ship::

    loop = Loop(my_agent)                       # any callable: input -> output
    run = loop.run("Summarise the sleep study")
    loop.fail(run, issue="missing_citation", must=["[source]"])
    verdict = loop.judge(my_new_agent)          # replayed on the record
    if verdict:
        loop.adopt(my_new_agent, verdict=verdict)

**Managed mode**: you give a configuration and a function that builds the
system from it. Changes are then data, so the loop can propose them, install
them, measure whether each one still earns its place, and take them out::

    loop = Loop(build=prompted(my_llm), config={"rules": []},
                proposers=[rules_from_feedback()])
    ...
    loop.learn()        # propose -> judge on held-out failures -> install
    loop.audit()        # switch each change off; drop the ones that stopped helping
    loop.rollback()     # undo the newest change

Every stage is an argument: ``checks`` score answers, ``gate`` decides,
``proposers`` suggest changes, ``store`` keeps the record. Each is a plain
callable or a small object, so any of them can be replaced without touching
the others.
"""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from erratica.core.checks import default_checks, score_case
from erratica.core.gates import ReplayGate, Trial, as_verdict
from erratica.core.proposers import Context, Proposer, as_changes
from erratica.core.store import open_store
from erratica.core.types import (
    APPROVED,
    FAILURE,
    IN_SAMPLE,
    AuditFinding,
    Case,
    Change,
    Check,
    Run,
    System,
    Verdict,
)
from erratica.learning.splits import split_cases

_MISSING = object()


def _mean(samples: Optional[Sequence[float]]) -> Optional[float]:
    return None if samples is None else sum(samples) / len(samples)


class Loop:
    """A system, the record of its mistakes, and the gate its changes pass.

    Arguments:
        system: the system, as a callable ``input -> output`` (judge mode).
        build: ``build(config) -> system`` (managed mode). Pass exactly one
            of ``system`` and ``build``.
        config: the starting configuration for ``build``. Use plain values
            (dicts, lists, text, numbers) so it can be stored and compared.
        baseline: the original system, when it differs from the one the loop
            starts with. A certified gate measures harm against it.
        checks: functions ``(case, output) -> score``. Defaults to the
            case's own ``must`` / ``must_not`` terms and its ``expect``.
        gate: ``gate(trial) -> Verdict``. Defaults to ``ReplayGate()``.
        proposers: functions ``(context) -> changes``, cheapest first.
        store: where the record lives. ``None`` keeps it in memory, a path
            opens one SQLite file, or pass your own store object.
        repeats: how many times each system answers each case when it is
            scored. Two or more lets a gate measure the system's own
            run-to-run noise and ask that a gain stand out from it. Raise
            it for a noisy model; set it to 1 to halve the cost when the
            system always answers the same way and your gate does not need
            a p-value.
        workers: how many cases are answered at once when scoring.
        holdout_fraction: the share of an issue's distinct failures kept
            away from the proposers and used to judge their changes.
        min_failures: distinct failures an issue needs before ``learn``
            works on it.
        triage: ``triage(issue, failures) -> bool`` to decide yourself which
            issues ``learn`` works on.
    """

    def __init__(
        self,
        system: Optional[System] = None,
        *,
        build: Optional[Callable[[Any], System]] = None,
        config: Any = None,
        baseline: Optional[System] = None,
        checks: Optional[Sequence[Check]] = None,
        gate: Optional[Callable[[Trial], Any]] = None,
        proposers: Sequence[Proposer] = (),
        store: Any = None,
        repeats: int = 2,
        workers: int = 1,
        holdout_fraction: float = 0.34,
        min_failures: int = 1,
        triage: Optional[Callable[[str, List[Case]], bool]] = None,
    ) -> None:
        if (system is None) == (build is None):
            raise TypeError("pass exactly one of system= (a callable) or build= (config -> callable)")
        if system is not None and config is not None:
            raise TypeError("config= only makes sense with build=")
        self.checks: List[Check] = list(checks) if checks is not None else default_checks()
        self.gate = gate if gate is not None else ReplayGate()
        self.proposers: List[Proposer] = list(proposers)
        self.repeats = max(1, int(repeats))
        self.workers = max(1, int(workers))
        self.holdout_fraction = holdout_fraction
        self.min_failures = max(1, int(min_failures))
        self.triage = triage
        self._events: Dict[str, List[Callable[[Any], None]]] = {}

        self._store = open_store(store)
        self._cases: List[Case] = [Case.from_dict(data, id=row_id) for row_id, data in self._store.rows("case")]
        self._installed: List[dict] = [
            {
                "id": row_id,
                "name": data.get("name", ""),
                "active": bool(data.get("active")),
                "change": Change.from_dict(data["change"]) if data.get("change") else None,
            }
            for row_id, data in self._store.rows("change")
        ]
        self.version = int(self._store.get_state("version", 0))
        self._reviewed: Dict[str, int] = dict(self._store.get_state("reviewed", {}) or {})
        if hasattr(self.gate, "restore"):
            self.gate.restore(self._store.get_state("gate"))

        self._build = build
        self._custom_baseline = baseline is not None
        self._base_config = copy.deepcopy(config if config is not None else {}) if build else None
        self._incumbent_scores: Dict[int, Optional[float]] = {}
        self._baseline_scores: Dict[int, Optional[float]] = {}
        if build is not None:
            self._config = self._fold()
            self._system = build(copy.deepcopy(self._config))
            self._baseline = baseline
        else:
            self._config = None
            self._system = system
            self._baseline = baseline if baseline is not None else system

    # ------------------------------------------------------------------
    # The system
    # ------------------------------------------------------------------

    @property
    def system(self) -> System:
        """The system as it is deployed now."""
        return self._system

    @property
    def config(self) -> Any:
        """A copy of the current configuration (managed mode)."""
        return copy.deepcopy(self._config)

    @property
    def managed(self) -> bool:
        return self._build is not None

    def run(self, input: Any, *, record: bool = True) -> Run:
        """Answer one input with the current system.

        The run is recorded so feedback can refer to it; pass
        ``record=False`` on a hot path and give ``fail`` / ``approve`` the
        input and output directly instead.
        """
        run = Run(input=input, output=self._system(input), version=self.version)
        if record:
            run.id = self._store.add("run", {"input": input, "output": run.output, "version": run.version})
        self._emit("run", run)
        return run

    def __call__(self, input: Any) -> Any:
        return self.run(input).output

    # ------------------------------------------------------------------
    # The record
    # ------------------------------------------------------------------

    def fail(
        self,
        what: Any,
        *,
        output: Any = _MISSING,
        issue: str = "general",
        expect: Any = None,
        must: Iterable[str] = (),
        must_not: Iterable[str] = (),
        note: str = "",
        meta: Optional[dict] = None,
    ) -> Case:
        """Record a mistake.

        ``what`` is a ``Run`` or the input itself. ``issue`` is a short
        label shared by mistakes of the same kind. Say what a good answer
        looks like with ``expect`` (the answer that was wanted), ``must`` /
        ``must_not`` (terms), or ``meta`` for your own checks; a failure
        that no check can score cannot be learned from.
        """
        return self._record(FAILURE, what, output, issue, expect, must, must_not, note, meta)

    def approve(
        self,
        what: Any,
        *,
        output: Any = _MISSING,
        issue: str = "general",
        expect: Any = None,
        must: Iterable[str] = (),
        must_not: Iterable[str] = (),
        note: str = "",
        meta: Optional[dict] = None,
    ) -> Case:
        """Record an answer someone accepted. Later changes must not break it.

        An approval protects an answer through the checks, so say what made
        it good: ``must`` / ``must_not`` terms, ``meta`` for your own
        checks, or rely on checks that apply to every answer. The approved
        text is kept in ``case.output``, but it is deliberately not treated
        as the one right answer: a different wording of a good answer is
        still good, and comparing wording cannot tell the two apart (see
        ``checks.stays_close_to_approved`` if you have a scorer that can).
        """
        return self._record(APPROVED, what, output, issue, expect, must, must_not, note, meta)

    def add(self, case: Case) -> Case:
        """Add a case built by hand (importing a test set, for example)."""
        if self._store.persistent:
            try:
                same = json.loads(json.dumps(case.input)) == case.input
            except (TypeError, ValueError):
                same = False
            if not same:
                raise TypeError(
                    "a recorded input is replayed later, so with a persistent store it must be "
                    "plain JSON (text, numbers, lists, dicts with text keys)"
                )
        case.id = self._store.add("case", case.to_dict())
        self._cases.append(case)
        self._emit("case", case)
        return case

    def _record(self, kind, what, output, issue, expect, must, must_not, note, meta) -> Case:
        if isinstance(what, Run):
            input, version = what.input, what.version
            output = what.output if output is _MISSING else output
        else:
            input, version = what, self.version
            output = None if output is _MISSING else output
        return self.add(
            Case(
                input=input,
                kind=kind,
                issue=issue,
                output=output,
                expect=expect,
                must=tuple(must),
                must_not=tuple(must_not),
                note=note,
                meta=dict(meta or {}),
                version=version,
            )
        )

    def cases(self, kind: Optional[str] = None, issue: Optional[str] = None) -> List[Case]:
        return [
            case
            for case in self._cases
            if (kind is None or case.kind == kind) and (issue is None or case.issue == issue)
        ]

    # ------------------------------------------------------------------
    # Judging and installing
    # ------------------------------------------------------------------

    def judge(
        self,
        candidate: Union[Change, System],
        *,
        issue: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Verdict:
        """Put a candidate through the gate without installing anything.

        ``candidate`` is a ``Change`` (managed mode) or a new system
        (a callable). With ``issue`` the target is that issue's failures;
        without it, every recorded failure. A candidate you wrote after
        reading the failures is judged ``in_sample``: that is weaker
        evidence than ``learn`` gets from held-out failures.
        """
        change = candidate if isinstance(candidate, Change) else None
        if change is not None:
            issue = issue if issue is not None else change.issue
            system = self._candidate(change)
            if isinstance(system, Verdict):
                return self._close(system)
        elif callable(candidate):
            system = candidate
        else:
            raise TypeError("a candidate is a Change or a callable")
        target = self.cases(FAILURE, issue)
        return self._judge(system, change, name, issue, target, IN_SAMPLE)

    def install(self, change: Change, *, issue: Optional[str] = None, force: bool = False) -> Verdict:
        """Judge a change and install it if the gate accepts it.

        ``force`` installs without asking the gate, and says so in the
        record. A certified gate does not allow it.
        """
        self._need_managed("install")
        if not isinstance(change, Change):
            raise TypeError("install takes a Change; to switch to a new callable use adopt")
        if force:
            self._refuse_if_certified("a forced install")
            system = self._candidate(change)
            if isinstance(system, Verdict):
                return self._close(system)
            verdict = self._close(
                Verdict(True, "installed without a gate (force=True)", change=change.name,
                        issue=issue or change.issue, gate="none")
            )
            self._install(change, system, verdict)
            return verdict
        issue = issue if issue is not None else change.issue
        system = self._candidate(change)
        if isinstance(system, Verdict):
            return self._close(system)
        verdict = self._judge(system, change, None, issue, self.cases(FAILURE, issue), IN_SAMPLE)
        if verdict.accepted:
            self._install(change, system, verdict)
        return verdict

    def adopt(
        self,
        system: System,
        *,
        name: str = "new system",
        issue: Optional[str] = None,
        verdict: Optional[Verdict] = None,
        force: bool = False,
    ) -> Verdict:
        """Judge a new system and, if accepted, make it the current one
        (judge mode). You still deploy it yourself; this keeps the loop and
        its record in step with what is deployed.

        Pass the ``verdict`` you already got from ``judge`` for this system
        to switch without judging it a second time.
        """
        if self.managed:
            raise TypeError("this loop builds its system from a configuration; use install(Change)")
        if verdict is not None:
            if not verdict.accepted or verdict.installed:
                raise ValueError("adopt needs an accepted verdict that has not been used yet")
            name = verdict.change or name
        elif force:
            self._refuse_if_certified("a forced adoption")
            verdict = self._close(Verdict(True, "adopted without a gate (force=True)", change=name,
                                          issue=issue, gate="none"))
        else:
            verdict = self.judge(system, issue=issue, name=name)
        if verdict.accepted:
            row_id = self._store.add("change", {"name": name, "active": True, "change": None,
                                                "verdict": verdict.id})
            self._installed.append({"id": row_id, "name": name, "active": True, "change": None})
            self._system = system
            self._bump()
            self._mark_installed(verdict)
        return verdict

    def learn(self, *, issues: Optional[Iterable[str]] = None, force: bool = False) -> List[Verdict]:
        """Work through the recorded failures and try to fix them.

        For each issue with failures this method has not seen before: the
        failures are split by input, the proposers write changes from one
        side, and the gate judges each change on the other. The first
        accepted change is installed and the issue is done. With too few
        distinct failures to split, changes are judged on the cases they
        came from and every verdict says ``in_sample``.

        ``force`` re-examines issues that have nothing new, which is worth
        doing after changing the model, the checks or the proposers.
        """
        self._need_managed("learn")
        if not self.proposers:
            raise ValueError("learn needs at least one proposer")
        wanted = set(issues) if issues is not None else None
        groups: Dict[str, List[Case]] = {}
        for case in self.cases(FAILURE):
            if wanted is None or case.issue in wanted:
                groups.setdefault(case.issue, []).append(case)

        verdicts: List[Verdict] = []
        for issue in sorted(groups):
            failures = groups[issue]
            newest = max(case.id or 0 for case in failures)
            if not force and newest <= int(self._reviewed.get(issue, 0)):
                continue
            if self.triage is not None:
                learnable = bool(self.triage(issue, failures))
            else:
                learnable = len({case.key for case in failures}) >= self.min_failures
            if not learnable:
                continue
            split = split_cases(failures, holdout_fraction=self.holdout_fraction, prompt_of=lambda c: c.key)
            context = Context(issue=issue, failures=list(split.train), config=self.config, validation=split.mode)
            verdicts.extend(self._work(context, split.holdout))
            # Marked only once the issue has been fully handled, so a pass
            # that dies halfway (a model server going down) is retried.
            self._reviewed[issue] = newest
            self._store.set_state("reviewed", self._reviewed)
        return verdicts

    def _work(self, context: Context, target: Sequence[Case]) -> List[Verdict]:
        verdicts: List[Verdict] = []
        for proposer in self.proposers:
            for change in as_changes(proposer(context)):
                change = change.with_issue(context.issue)
                system = self._candidate(change, quiet=True)
                if system is None:
                    continue
                if isinstance(system, Verdict):
                    verdicts.append(self._close(system))
                    context.rejected.append(change.name)
                    continue
                verdict = self._judge(system, change, None, context.issue, target, context.validation)
                verdicts.append(verdict)
                if verdict.accepted:
                    self._install(change, system, verdict)
                    return verdicts
                context.rejected.append(change.name)
        return verdicts

    # ------------------------------------------------------------------
    # Keeping the system honest afterwards
    # ------------------------------------------------------------------

    def audit(self, *, min_cost: float = 0.0) -> List[AuditFinding]:
        """Switch each installed change off in turn and replay the record.

        A change whose removal costs no more than ``min_cost`` in mean
        score is taken out: it was made redundant by a later change, or it
        only looked good on the few cases it was judged on. Newest first.
        """
        self._need_managed("audit")
        self._refuse_if_certified("an audit")
        findings: List[AuditFinding] = []
        for entry in reversed([e for e in self._installed if e["active"] and e["change"] is not None]):
            name = entry["change"].name
            without_config = self._fold(skip=entry["id"])
            if without_config == self._config:
                self._deactivate(entry, "audit: it no longer changes the configuration")
                findings.append(AuditFinding(name, 0.0, 0, True, "no longer changes the configuration"))
                continue
            try:
                without = self._build(copy.deepcopy(without_config))
            except Exception as exc:
                findings.append(AuditFinding(name, None, 0, False, f"could not build without it: {exc}"))
                continue
            with_scores = self._score(self._system, self._cases, self._incumbent_cache(), strict=True)
            without_scores = self._score(without, self._cases, {}, strict=False)
            pairs = [(a, b) for a, b in zip(with_scores, without_scores) if a is not None and b is not None]
            if not pairs:
                findings.append(AuditFinding(name, None, 0, False, "no check applies to any recorded case"))
                continue
            cost = sum(a - b for a, b in pairs) / len(pairs)
            if cost <= min_cost:
                self._deactivate(entry, f"audit: removal costs {cost:+.3f}")
                findings.append(AuditFinding(name, cost, len(pairs), True, "removing it costs nothing"))
            else:
                findings.append(AuditFinding(name, cost, len(pairs), False, "still earning its place"))
        return findings

    def rollback(self, which: Union[int, str, None] = None, *, reason: str = "manual rollback") -> bool:
        """Undo an installed change: the newest by default, or one chosen
        by its id or name (see ``history``). Returns False if there is no
        such active change."""
        self._need_managed("rollback")
        active = [e for e in self._installed if e["active"] and e["change"] is not None]
        if which is None:
            chosen = active[-1:]
        elif isinstance(which, int):
            chosen = [e for e in active if e["id"] == which]
        else:
            chosen = [e for e in active if e["change"].name == which][-1:]
        if not chosen:
            return False
        if getattr(self.gate, "certified", False) and chosen[0] is not active[-1]:
            raise ValueError(
                "with a certified gate only the newest change can be rolled back: removing one "
                "from underneath another would leave a system that was never certified"
            )
        self._deactivate(chosen[0], reason)
        return True

    def history(self) -> List[dict]:
        """Every change ever installed, oldest first, and whether it is still active."""
        return [
            {
                "id": entry["id"],
                "name": entry["change"].name if entry["change"] is not None else entry["name"],
                "kind": entry["change"].kind if entry["change"] is not None else "system",
                "issue": entry["change"].issue if entry["change"] is not None else None,
                "active": entry["active"],
                "ops": [dict(op) for op in entry["change"].ops] if entry["change"] is not None else None,
            }
            for entry in self._installed
        ]

    def verdicts(self) -> List[dict]:
        """Every verdict ever given, accepted or not, oldest first."""
        return [dict(data, id=row_id) for row_id, data in self._store.rows("verdict")]

    def score(self, system: Optional[System] = None) -> dict:
        """Mean score on the recorded failures and on the approved answers,
        for the current system or for one you pass. Handy in a CI job."""
        chosen = system if system is not None else self._system
        cache = self._incumbent_cache() if system is None else {}
        report: Dict[str, Any] = {"version": self.version}
        for kind, label in ((FAILURE, "failures"), (APPROVED, "approved")):
            scores = [s for s in self._score(chosen, self.cases(kind), cache, strict=True) if s is not None]
            report[label] = {"score": sum(scores) / len(scores) if scores else None, "cases": len(scores)}
        return report

    def measure(self, cases: Sequence[Case], system: Optional[System] = None) -> dict:
        """Score cases that are not part of the record (a test set kept
        aside, say) with the current system or one you pass. Nothing is
        stored, and the cases are never shown to a proposer."""
        chosen = system if system is not None else self._system
        errors: List[str] = []
        scores = [_mean(self._answer(chosen, case, errors)) for case in cases]
        scored = [value for value in scores if value is not None]
        return {
            "score": sum(scored) / len(scored) if scored else None,
            "cases": len(scored),
            "scores": scores,
            "errors": len(errors),
        }

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def on(self, event: str, handler: Callable[[Any], None]) -> "Loop":
        """Call ``handler`` on ``"run"``, ``"case"``, ``"verdict"``,
        ``"install"`` or ``"rollback"``. Returns the loop, so calls chain."""
        self._events.setdefault(event, []).append(handler)
        return self

    def _emit(self, event: str, payload: Any) -> None:
        for handler in self._events.get(event, ()):
            handler(payload)

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "Loop":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _need_managed(self, what: str) -> None:
        if not self.managed:
            raise TypeError(f"{what} needs a loop created with build= and config=")

    def _refuse_if_certified(self, what: str) -> None:
        if getattr(self.gate, "certified", False):
            raise ValueError(f"{what} would change the system without a certificate; the gate is certified")

    def _fold(self, skip: Optional[int] = None) -> Any:
        """The configuration with every active change applied, in order."""
        config = copy.deepcopy(self._base_config)
        for entry in self._installed:
            if entry["active"] and entry["change"] is not None and entry["id"] != skip:
                config = entry["change"].apply(config)
        return config

    def _candidate(self, change: Change, *, quiet: bool = False) -> Any:
        """The system a change would produce, or a rejecting Verdict if it
        cannot be applied or built. A change that alters nothing gets a
        rejecting Verdict too, or None when ``quiet`` (a proposer repeating
        itself is not worth a line in the record)."""
        self._need_managed("a Change")
        try:
            config = change.apply(self._config)
        except (TypeError, ValueError) as exc:
            return Verdict(False, f"could not be applied: {exc}", change=change.name, issue=change.issue,
                           gate="none")
        if config == self._config:
            if quiet:
                return None
            return Verdict(False, "changes nothing in the configuration", change=change.name,
                           issue=change.issue, gate="none")
        try:
            return self._build(copy.deepcopy(config))
        except Exception as exc:
            return Verdict(False, f"could not be built: {exc}", change=change.name, issue=change.issue,
                           gate="none")

    def _baseline_system(self) -> System:
        if self._baseline is None:
            self._baseline = self._build(copy.deepcopy(self._base_config))
        return self._baseline

    def _incumbent_is_baseline(self) -> bool:
        if self._custom_baseline:
            return self._system is self._baseline
        return self._config == self._base_config if self.managed else self._system is self._baseline

    def _incumbent_cache(self) -> Dict[int, Optional[float]]:
        return self._incumbent_scores

    def _answer(self, system: System, case: Case, errors: List[str]) -> Optional[Tuple[float, ...]]:
        """One score per repeat, or None when no check applies to the case."""
        values: List[float] = []
        for _ in range(self.repeats):
            try:
                output = system(case.input)
            except Exception as exc:  # a crash is a failed answer
                errors.append(f"{type(exc).__name__}: {exc}")
                values.append(0.0)
                continue
            value = score_case(case, output, self.checks)
            if value is None:
                return None
            values.append(value)
        return tuple(values)

    def _score(self, system: System, cases: Sequence[Case], cache: dict, *, strict: bool) -> List[Optional[float]]:
        """Mean score per case (None where no check applies)."""
        return [_mean(samples) for samples in self._samples(system, cases, cache, strict=strict)]

    def _samples(
        self,
        system: System,
        cases: Sequence[Case],
        cache: Dict[int, Optional[Tuple[float, ...]]],
        *,
        strict: bool,
        errors: Optional[List[str]] = None,
    ) -> List[Optional[Tuple[float, ...]]]:
        """Every repeat's score for ``cases``, answering only the ones not in ``cache``.

        ``strict`` is for systems that are supposed to work (the incumbent,
        the baseline): if every single call raises, the cause is most
        likely outside the system (a server that is down), and that is
        reported instead of being scored as a wall of zeros.
        """
        errors = errors if errors is not None else []
        todo = [case for case in cases if case.id not in cache]
        before = len(errors)
        if todo:
            if self.workers > 1 and len(todo) > 1:
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    results = list(pool.map(lambda case: self._answer(system, case, errors), todo))
            else:
                results = [self._answer(system, case, errors) for case in todo]
            failed = len(errors) - before
            if strict and failed == len(todo) * self.repeats:
                raise RuntimeError(
                    f"the system raised on every one of {failed} replayed calls; last error: {errors[-1]}"
                )
            for case, value in zip(todo, results):
                cache[case.id] = value
        return [cache[case.id] for case in cases]

    def _judge(
        self,
        system: System,
        change: Optional[Change],
        name: Optional[str],
        issue: Optional[str],
        target: Sequence[Case],
        validation: str,
    ) -> Verdict:
        candidate_scores: Dict[int, Optional[Tuple[float, ...]]] = {}
        candidate_errors: List[str] = []

        def scorer(role: str, cases: Sequence[Case]) -> List[Optional[Tuple[float, ...]]]:
            if role == "candidate":
                return self._samples(system, cases, candidate_scores, strict=False, errors=candidate_errors)
            if role == "incumbent":
                return self._samples(self._system, cases, self._incumbent_cache(), strict=True)
            if role == "baseline":
                return self._samples(self._baseline_system(), cases, self._baseline_scores, strict=True)
            raise ValueError(f"unknown role {role!r}")

        trial = Trial(
            candidate=system,
            incumbent=self._system,
            baseline=self._baseline_system,
            target=target,
            approved=self.cases(APPROVED),
            others=[] if issue is None else [c for c in self.cases(FAILURE) if c.issue != issue],
            validation=validation,
            scorer=scorer,
            change=change,
            issue=issue,
            name=name,
            incumbent_is_baseline=self._incumbent_is_baseline(),
        )
        try:
            verdict = as_verdict(self.gate(trial), trial, self.gate)
        finally:
            # Budget and samples are spent even if the gate is interrupted.
            if hasattr(self.gate, "state"):
                self._store.set_state("gate", self.gate.state())
        verdict.validation = validation
        if candidate_errors:
            verdict.evidence["candidate_errors"] = {"count": len(candidate_errors), "last": candidate_errors[-1]}
        if change is not None:
            verdict.evidence["ops"] = [dict(op) for op in change.ops]
        return self._close(verdict)

    def _close(self, verdict: Verdict) -> Verdict:
        verdict.id = self._store.add("verdict", verdict.to_dict())
        self._emit("verdict", verdict)
        return verdict

    def _install(self, change: Change, system: System, verdict: Verdict) -> None:
        row_id = self._store.add(
            "change", {"name": change.name, "active": True, "change": change.to_dict(), "verdict": verdict.id}
        )
        self._installed.append({"id": row_id, "name": change.name, "active": True, "change": change})
        self._config = self._fold()
        self._system = system
        self._bump()
        self._mark_installed(verdict)

    def _mark_installed(self, verdict: Verdict) -> None:
        verdict.installed, verdict.version = True, self.version
        self._store.put("verdict", verdict.id, verdict.to_dict())
        self._emit("install", verdict)

    def _deactivate(self, entry: dict, reason: str) -> None:
        config = self._fold(skip=entry["id"])
        system = self._build(copy.deepcopy(config))  # build first: a failure here changes nothing
        entry["active"] = False
        change = entry["change"]
        self._store.put(
            "change",
            entry["id"],
            {"name": change.name, "active": False, "change": change.to_dict(), "rolled_back": reason},
        )
        self._config, self._system = config, system
        self._bump()
        self._emit("rollback", {"id": entry["id"], "name": change.name, "reason": reason, "version": self.version})

    def _bump(self) -> None:
        self.version += 1
        self._store.set_state("version", self.version)
        self._incumbent_scores = {}
