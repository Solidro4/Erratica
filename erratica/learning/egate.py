"""A promotion gate with a statistical guarantee.

The default gate in ``promoter.py`` is a heuristic: a patch is promoted when
its mean score rises by a fixed amount on a handful of held-out cases. That
is fine for getting started and says nothing about how often it is wrong.
Run it for hundreds of patches and some regressions will get through by
chance.

This module is the rigorous alternative. It certifies:

    With probability at least 1 - delta, over an unlimited sequence of
    candidate patches, every patch that is ever promoted
      (a) leaves the true mean score of each protected behaviour it was
          tested on less than ``margin`` below the starting system's, and
      (b) truly raises the mean score of its task over the system it
          replaced (when a task test is required).

The full statement, its assumptions, and the proof are in
``docs/THEORY.md``. In outline:

**One test.** For each candidate and each protected behaviour, paired score
differences ``candidate - baseline`` on fresh evaluation samples feed a
betting e-process (Waudby-Smith and Ramdas, "Estimating means of bounded
random variables by betting"). Under the null hypothesis "the candidate is
worse than baseline by at least ``margin``" the e-process is a nonnegative
supermartingale, so by Ville's inequality the chance it ever reaches
``1 / alpha`` is at most ``alpha``, no matter when we stop looking.

**The bet.** How much to bet on each sample decides how many samples a test
costs, never whether it is valid. The process here hedges two strategies
half and half: a mixture over a grid of fixed bets, which is guaranteed to
end within ``log(2 * K)`` of the best fixed bet on the grid in hindsight,
and a follow-the-leader bettor that plays the grid bet that would have done
best so far. Any convex combination of valid e-processes is a valid
e-process.

**Many tests per candidate.** A candidate is promoted only if EVERY test
(each protected behaviour, and the task) crosses its threshold. A candidate
that fails (a) or (b) fails at least one of them, and it can only be
promoted if that one test errs. So no Bonferroni split is needed: each test
runs at the candidate's full level.

**Many candidates.** The k-th candidate to enter the gate is given a budget
``alpha_k`` from a series that sums to ``delta``. The default series decays
like ``1 / (k log^2 k)``, so the evidence a test needs grows like ``log k``.

**Recycled budget (online fallback).** When a candidate is promoted, the
level it was tested at is passed to the next candidate on top of that
candidate's own budget (Tian and Ramdas, "Online control of the familywise
error rate"). The first wrongly promoted candidate can only have inherited
budget from an unbroken run of correctly promoted ones, and those runs do
not overlap, so the levels at which a first error can occur still sum to at
most ``delta``. This needs no independence between tests.

**Stopping early on bad candidates.** Beside each test runs a mirror image
that gathers evidence the candidate is worse than the reference at all (for
a protected behaviour: worse than baseline; for the task: worse than the
incumbent). When that evidence reaches ``1 / futility`` the candidate is
rejected at once. Stopping early can never cause a wrong promotion, and a
candidate that is truly no worse is stopped this way with probability at
most ``futility``, so the cost in power is bounded too.

**Reusing samples (optional, ``reuse=True``).** Evaluation samples are
normally used once. With reuse, one batch serves every candidate until one
of them is promoted; then the batch is retired. This is valid as long as
whatever writes patches learns nothing from a rejection except that it
happened: along a run of rejections the candidates are then the ones that
would have been proposed anyway, a list fixed in advance, and testing a
fixed list on one batch is an ordinary union bound. The budget becomes
two-level, ``delta * w(batch) * g(position in batch)``. Details of rejected
tests must stay sealed until the batch is retired. See Theorem 7 in
``docs/THEORY.md``.

What the guarantee assumes
--------------------------
1. Scores are bounded in [0, 1].
2. For each behaviour, evaluation samples come from a fixed distribution,
   and a sample used to test a candidate is independent of that candidate
   and of everything that came before. Using every sample exactly once
   gives this (see ``evalstream.py``).
3. The candidate being tested is the system that would be deployed.
4. The guarantee is about the scores. If the scorer does not measure what
   you care about, neither does the certificate.

The margin cannot be zero. A patch that leaves a protected behaviour exactly
unchanged gives no evidence that it is better, so with ``margin = 0`` no
neutral patch could ever be promoted. The margin sets the price: a harmless
patch whose paired differences are all zero costs about
``log(1 / alpha) / margin`` samples, and no valid test can do better than
``log(1 / alpha) / log(1 / (1 - margin))`` (see ``expected_samples``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

PROMOTE = "promote"
REJECT = "reject"
CONTINUE = "continue"

# Upper bound on sum_{n >= 2} 1 / (n * ln(n)^2) = 2.10974...
_LOG_SERIES = 2.11

GRID_SIZE = 12
GRID_RATIO = 0.66
PRIOR_WEIGHT = 0.5
BET_CAP = 0.9


def alpha_for(k: int, delta: float, schedule: str = "log") -> float:
    """Error budget of the k-th candidate (k = 1, 2, ...). Sums to at most delta.

    ``"log"`` spends ``delta / (C * (k+1) * ln(k+1)^2)``, the slowest decay
    that still sums, so late candidates stay affordable. ``"quadratic"`` is
    the textbook ``6 * delta / (pi * k)^2``: slightly more generous for the
    first four candidates and much stingier afterwards.
    """
    if k < 1:
        raise ValueError("k starts at 1")
    if schedule == "quadratic":
        return 6.0 * delta / (math.pi * k) ** 2
    if schedule == "log":
        n = k + 1
        return delta / (_LOG_SERIES * n * math.log(n) ** 2)
    raise ValueError(f"unknown schedule '{schedule}'")


def batch_weight(batch: int) -> float:
    """Share of delta given to the batch-th evaluation batch (1, 2, ...)."""
    if batch < 1:
        raise ValueError("batches are numbered from 1")
    n = batch + 1
    return 1.0 / (_LOG_SERIES * n * math.log(n) ** 2)


def position_weight(position: int) -> float:
    """Share of a batch's budget for its position-th candidate. Sums to 1."""
    if position < 1:
        raise ValueError("positions are numbered from 1")
    return 1.0 / (position * (position + 1))


def alpha_for_position(batch: int, position: int, delta: float) -> float:
    """Own error budget of a candidate when samples are reused within a batch."""
    return delta * batch_weight(batch) * position_weight(position)


def bet_grid(lower: float, size: int = GRID_SIZE, ratio: float = GRID_RATIO) -> list:
    """Candidate bets, largest first. The largest keeps wealth positive."""
    largest = BET_CAP / abs(lower)
    return [largest * ratio**index for index in range(size)]


def _log_mean_exp(values: Sequence[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values) / len(values))


class BettingEProcess:
    """One-sided test of H0: E[x] <= 0 for observations in [lower, upper].

    The wealth is an equal mix of two bettors over the same grid of bets:

    - *mixture*: holds every grid bet at once, so it is never more than
      ``log(K)`` behind the best fixed grid bet in hindsight;
    - *leader*: each round plays the single grid bet with the best record
      so far (regularised by half a pseudo-observation at each end of the
      range, so it starts cautious). It learns faster when the data are
      steady.

    Each bettor's wealth is a nonnegative supermartingale under H0, so the
    mix is too.
    """

    def __init__(self, lower: float = -1.0, upper: Optional[float] = None) -> None:
        if lower >= 0:
            raise ValueError("lower must be negative")
        self.lower = lower
        self.upper = lower + 2.0 if upper is None else upper
        self.bets = bet_grid(lower)
        self._prior = [
            PRIOR_WEIGHT * (math.log1p(bet * self.lower) + math.log1p(bet * self.upper))
            for bet in self.bets
        ]
        self._fixed = [0.0] * len(self.bets)
        self._leader = 0.0
        self.observations: list = []
        self.log_wealth = 0.0
        self.max_log_wealth = 0.0

    @property
    def n(self) -> int:
        return len(self.observations)

    def next_bet(self) -> float:
        """The leader's bet for the next observation (past data only)."""
        scores = [fixed + prior for fixed, prior in zip(self._fixed, self._prior)]
        return self.bets[scores.index(max(scores))]

    def update(self, x: float) -> float:
        if not (self.lower - 1e-9 <= x <= self.upper + 1e-9):
            raise ValueError(f"observation {x} is outside [{self.lower}, {self.upper}]")
        x = min(self.upper, max(self.lower, x))
        self._leader += math.log1p(self.next_bet() * x)
        self._fixed = [fixed + math.log1p(bet * x) for fixed, bet in zip(self._fixed, self.bets)]
        self.log_wealth = _log_mean_exp([_log_mean_exp(self._fixed), self._leader])
        self.max_log_wealth = max(self.max_log_wealth, self.log_wealth)
        self.observations.append(x)
        return self.log_wealth

    def crossed(self, alpha: float) -> bool:
        """Has the wealth ever reached 1 / alpha?"""
        return self.max_log_wealth >= -math.log(alpha)


def lower_confidence_bound(
    observations: Sequence[float],
    alpha: float,
    *,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    """A value the true mean exceeds with probability at least 1 - alpha.

    Valid at whatever moment sampling stopped. For each hypothetical mean
    ``m`` a grid-mixture e-process tests "the mean is at most m"; the bound
    is the largest ``m`` that test rejects. Bets are scaled by the distance
    from ``m`` to the bottom of the range, which makes the evidence fall
    steadily as ``m`` rises, so bisection finds the exact edge.

    This is a pure mixture, without the follow-the-leader half the gate's
    own test uses, so right after a test crosses its threshold the bound
    can sit slightly below the value that test just ruled out.
    """
    if not observations:
        return lower
    fractions = [BET_CAP * GRID_RATIO**index for index in range(GRID_SIZE)]
    threshold = -math.log(alpha)
    shifted = [x - lower for x in observations]  # all >= 0

    def rejected(m: float) -> bool:
        span = m - lower
        logs = [
            sum(math.log1p(fraction * (value / span - 1.0)) for value in shifted)
            for fraction in fractions
        ]
        return _log_mean_exp(logs) >= threshold

    low, high = lower, upper
    if not rejected(lower + 1e-9 * (upper - lower)):
        return lower
    for _ in range(30):
        middle = (low + high) / 2.0
        if rejected(middle):
            low = middle
        else:
            high = middle
    return low


def best_growth(outcomes: Sequence[float], probabilities: Sequence[float], lower: float) -> float:
    """Largest expected log-growth per sample any allowed fixed bet achieves
    on a known distribution of observations (the Kelly rate)."""

    def growth(bet: float) -> float:
        return sum(p * math.log1p(bet * x) for x, p in zip(outcomes, probabilities) if p > 0)

    low, high = 0.0, BET_CAP / abs(lower)
    for _ in range(80):
        first = low + (high - low) / 3.0
        second = high - (high - low) / 3.0
        if growth(first) < growth(second):
            low = first
        else:
            high = second
    return max(0.0, growth((low + high) / 2.0))


def expected_samples(
    alpha: float,
    margin: float,
    *,
    gains: float = 0.0,
    losses: float = 0.0,
) -> dict:
    """Roughly how many samples one protected-behaviour test will cost.

    Models the paired difference ``candidate - baseline`` as +1 with
    probability ``gains``, -1 with probability ``losses``, and 0 otherwise
    (exact for pass/fail scorers). Returns:

    - ``typical``: ``log(1 / alpha)`` divided by the best achievable growth
      rate with this module's bets. Real tests cost somewhat more while the
      bettor is still learning the rate, most noticeably when the threshold
      is small.
    - ``floor``: for a patch that changes nothing (no gains, no losses), the
      fewest samples ANY valid test of bounded scores could need,
      ``log(1 / alpha) / log(1 / (1 - margin))``. None otherwise.

    The conservation law this exposes: certifying candidate k costs about
    ``(log(1 / delta) + log k) / growth`` samples, where growth is of order
    ``margin`` for a quiet behaviour and ``(margin + mean change)^2 /
    (2 * variance)`` for a noisy one. Safety (delta), tolerance (margin),
    and the number of patches a sample budget can certify trade off
    against each other, and nothing else is free.
    """
    if not 0 < margin < 1:
        raise ValueError("margin must be in (0, 1)")
    if gains < 0 or losses < 0 or gains + losses > 1:
        raise ValueError("gains and losses must be probabilities summing to at most 1")
    outcomes = [1.0 + margin, -1.0 + margin, margin]
    probabilities = [gains, losses, 1.0 - gains - losses]
    rate = best_growth(outcomes, probabilities, lower=-1.0 + margin)
    need = -math.log(alpha)
    quiet = gains == 0 and losses == 0
    return {
        "growth_per_sample": rate,
        "typical": math.inf if rate <= 0 else need / rate,
        "floor": need / -math.log(1.0 - margin) if quiet else None,
    }


@dataclass
class GateTest:
    """The test of one candidate patch. Feed it paired scores; read ``decision``."""

    k: int
    alpha: float
    margin: float
    max_samples: int
    behaviours: tuple
    require_improvement: bool = True
    # Portion of ``alpha`` inherited from the previous, promoted candidate.
    inherited: float = 0.0
    # Reject once there is evidence at this level that the candidate is
    # worse than its reference at all (None: always run to ``max_samples``).
    futility: Optional[float] = 0.02
    delta: Optional[float] = None
    schedule: str = "log"
    # Which evaluation batch this test read, and its place among the
    # candidates of that batch. Only meaningful when samples are reused.
    reuse: bool = False
    batch: int = 1
    position: int = 1
    _protected: dict = field(default_factory=dict)
    _step: Optional[BettingEProcess] = None
    _worse: dict = field(default_factory=dict)
    _step_worse: Optional[BettingEProcess] = None

    def __post_init__(self) -> None:
        for behaviour in self.behaviours:
            # x = (candidate - baseline) + margin lies in [-1 + margin, 1 + margin].
            self._protected[behaviour] = BettingEProcess(
                lower=-1.0 + self.margin,
                upper=1.0 + self.margin,
            )
            # The mirror image: evidence that the candidate is worse at all.
            self._worse[behaviour] = BettingEProcess(lower=-1.0, upper=1.0)
        if self.require_improvement:
            self._step = BettingEProcess(lower=-1.0, upper=1.0)
            self._step_worse = BettingEProcess(lower=-1.0, upper=1.0)

    def add_protected(self, behaviour: str, candidate: float, baseline: float) -> None:
        """One fresh sample of a protected behaviour, scored for the candidate
        and for the ORIGINAL baseline (not the current model: anchoring to
        the starting point is what stops small losses from adding up)."""
        _check_score(candidate)
        _check_score(baseline)
        self._protected[behaviour].update(candidate - baseline + self.margin)
        self._worse[behaviour].update(baseline - candidate)

    def add_task(self, candidate: float, incumbent: float) -> None:
        """One fresh sample of the task being improved, scored for the
        candidate and for the currently deployed model."""
        if self._step is None:
            raise ValueError("this test was created with require_improvement=False")
        _check_score(candidate)
        _check_score(incumbent)
        self._step.update(candidate - incumbent)
        self._step_worse.update(incumbent - candidate)

    def gave_up(self) -> list:
        """Tests stopped because the evidence points the other way."""
        if self.futility is None:
            return []
        stopped = [
            name
            for name, process in self._worse.items()
            if process.crossed(self.futility) and not self._protected[name].crossed(self.alpha)
        ]
        if (
            self._step_worse is not None
            and self._step_worse.crossed(self.futility)
            and not self._step.crossed(self.alpha)
        ):
            stopped.append(None)
        return stopped

    def settled(self, behaviour: Optional[str]) -> bool:
        """True once a test needs no more samples (``None`` means the task test)."""
        process = self._step if behaviour is None else self._protected[behaviour]
        if process is None:
            return True
        return process.crossed(self.alpha) or process.n >= self.max_samples

    @property
    def decision(self) -> str:
        processes = list(self._protected.values())
        if self._step is not None:
            processes.append(self._step)
        if all(process.crossed(self.alpha) for process in processes):
            return PROMOTE
        if self.gave_up():
            return REJECT
        if any(
            process.n >= self.max_samples and not process.crossed(self.alpha)
            for process in processes
        ):
            return REJECT
        return CONTINUE

    def certificate(self) -> dict:
        """Everything a third party needs to check the decision.

        ``differences`` are the raw paired score differences in the order
        they were observed; ``verify_certificate`` replays them.

        ``certified_above`` is the statement the decision rests on: the
        true mean change in this score exceeds that value (minus the margin
        for a protected behaviour, zero for the task). ``change_at_least``
        is an additional lower confidence bound computed from the same
        samples by a different valid test; it is a separate statement, also
        at level ``alpha``, and early on it can be the looser of the two.
        """
        threshold = -math.log(self.alpha)

        def describe(process: BettingEProcess, shift: float) -> dict:
            crossed = process.crossed(self.alpha)
            differences = [x - shift for x in process.observations]
            return {
                "samples": process.n,
                "max_log_e": process.max_log_wealth,
                "crossed": crossed,
                "certified_above": -shift if crossed else None,
                "observed_mean_change": sum(differences) / process.n if process.n else None,
                "change_at_least": lower_confidence_bound(differences, self.alpha),
                "differences": differences,
            }

        gave_up = self.gave_up()
        return {
            "k": self.k,
            "delta": self.delta,
            "schedule": self.schedule,
            "reuse": self.reuse,
            "batch": self.batch,
            "position": self.position,
            "alpha": self.alpha,
            "alpha_inherited": self.inherited,
            "margin": self.margin,
            "max_samples": self.max_samples,
            "futility": self.futility,
            "log_threshold": threshold,
            "decision": self.decision,
            "stopped_early_on": ["task" if name is None else name for name in gave_up],
            "protected": {
                name: describe(process, self.margin) for name, process in self._protected.items()
            },
            "task": describe(self._step, 0.0) if self._step is not None else None,
        }


class EGate:
    """Hands out error budget to candidates and remembers how much is spent.

    Only call ``begin`` for candidates worth certifying. Budget is spent per
    candidate that enters the gate, so screening out obvious losers with a
    cheap uncertified check first keeps ``k`` small and later tests cheap.

    ``state()`` returns what must be saved between sessions; pass it back
    as ``EGate(..., **state)``. The guarantee covers the whole sequence only
    if the count keeps going.
    """

    def __init__(
        self,
        behaviours,
        *,
        delta: float = 0.05,
        margin: float = 0.05,
        max_samples: int = 2000,
        schedule: str = "log",
        recycle: bool = True,
        futility: Optional[float] = 0.02,
        reuse: bool = False,
        tested: int = 0,
        carry: float = 0.0,
        batch: int = 1,
        position: int = 0,
    ) -> None:
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if not 0 < margin < 1:
            raise ValueError("margin must be in (0, 1)")
        if futility is not None and not 0 < futility < 1:
            raise ValueError("futility must be in (0, 1) or None")
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        self.behaviours = tuple(behaviours)
        if len(set(self.behaviours)) != len(self.behaviours):
            raise ValueError("behaviour names must be unique")
        self.delta = delta
        self.margin = margin
        self.max_samples = max_samples
        self.schedule = schedule
        self.recycle = recycle
        self.futility = futility
        self.reuse = reuse
        self.tested = tested
        # The evaluation batch in use and how many candidates have read it.
        self.batch = batch
        self.position = position
        self._carry = carry
        self._last: Optional[GateTest] = None
        alpha_for(1, delta, schedule)  # validates the schedule name

    def _inheritance(self) -> float:
        if not self.recycle:
            return 0.0
        if self._last is not None:
            return self._last.alpha if self._last.decision == PROMOTE else 0.0
        return self._carry

    def begin(
        self,
        *,
        behaviours: Optional[Sequence[str]] = None,
        require_improvement: bool = True,
    ) -> GateTest:
        """Start testing the next candidate.

        ``behaviours`` may name a subset of the protected behaviours when
        the candidate provably cannot affect the others (for example a rule
        scoped to one route). The untested ones keep the certificate they
        already had.
        """
        chosen = tuple(self.behaviours if behaviours is None else behaviours)
        unknown = set(chosen) - set(self.behaviours)
        if unknown:
            raise ValueError(f"not protected behaviours of this gate: {sorted(unknown)}")
        if not chosen and not require_improvement:
            raise ValueError("a test needs at least one behaviour or the task test")
        inherited = self._inheritance()
        if self._last is not None and self._last.decision == PROMOTE:
            # A promotion retires the batch its candidate was tested on.
            self.batch += 1
            self.position = 0
        self.tested += 1
        self.position += 1
        if self.reuse:
            own = alpha_for_position(self.batch, self.position, self.delta)
            if self.position > 1:
                inherited = 0.0
        else:
            own = alpha_for(self.tested, self.delta, self.schedule)
        test = GateTest(
            k=self.tested,
            alpha=own + inherited,
            margin=self.margin,
            max_samples=self.max_samples,
            behaviours=chosen,
            require_improvement=require_improvement,
            inherited=inherited,
            futility=self.futility,
            delta=self.delta,
            schedule=self.schedule,
            reuse=self.reuse,
            batch=self.batch,
            position=self.position,
        )
        self._last = test
        return test

    def retire_batch(self) -> None:
        """Stop reusing the current evaluation batch.

        Call this before revealing details of rejected tests, or whenever
        the cached samples are lost. The next candidate starts a new batch
        with fresh samples and the next share of the budget.
        """
        promoted = self._last is not None and self._last.decision == PROMOTE
        if self.position > 0 and not promoted:
            self.batch += 1
            self.position = 0

    def state(self) -> dict:
        promoted = self._last is not None and self._last.decision == PROMOTE
        return {
            "tested": self.tested,
            "carry": self._inheritance(),
            "batch": self.batch + (1 if promoted else 0),
            "position": 0 if promoted else self.position,
        }


def verify_certificate(certificate: dict) -> list:
    """Recompute a certificate from its raw differences.

    Returns a list of problems; an empty list means the recorded decision
    follows from the recorded data. This checks the arithmetic of one test.
    Whether the inherited budget was legitimate depends on the previous
    candidate, which ``verify_chain`` checks.
    """
    problems: list = []
    try:
        k = int(certificate["k"])
        delta = float(certificate["delta"])
        if certificate.get("reuse"):
            own = alpha_for_position(int(certificate["batch"]), int(certificate["position"]), delta)
        else:
            own = alpha_for(k, delta, certificate.get("schedule", "log"))
        alpha = float(certificate["alpha"])
        inherited = float(certificate.get("alpha_inherited", 0.0))
        margin = float(certificate["margin"])
    except (KeyError, TypeError, ValueError) as exc:
        return [f"certificate is missing or has a malformed field: {exc}"]
    if certificate.get("reuse") and int(certificate["position"]) > 1 and inherited > 0:
        problems.append("only the first candidate of a batch may inherit budget")

    if not math.isclose(alpha, own + inherited, rel_tol=1e-9, abs_tol=1e-15):
        problems.append(f"alpha {alpha} is not own budget {own} plus inherited {inherited}")

    crossings: list = []

    def replay(name: str, entry: dict, lower: float, shift: float) -> None:
        process = BettingEProcess(lower=lower, upper=lower + 2.0)
        try:
            for difference in entry["differences"]:
                if not -1.0 - 1e-9 <= difference <= 1.0 + 1e-9:
                    raise ValueError(f"difference {difference} is outside [-1, 1]")
                process.update(difference + shift)
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"{name}: cannot replay the recorded differences: {exc}")
            crossings.append(False)
            return
        crossed = process.crossed(alpha)
        crossings.append(crossed)
        if crossed != bool(entry.get("crossed")):
            problems.append(f"{name}: recorded crossed={entry.get('crossed')} but replay gives {crossed}")
        if process.n != entry.get("samples"):
            problems.append(f"{name}: {entry.get('samples')} samples recorded, {process.n} differences stored")

    for name, entry in (certificate.get("protected") or {}).items():
        replay(name, entry, -1.0 + margin, margin)
    if certificate.get("task") is not None:
        replay("task", certificate["task"], -1.0, 0.0)
    if not crossings:
        problems.append("certificate contains no test")

    if certificate.get("decision") == PROMOTE and not (crossings and all(crossings)):
        problems.append("decision is 'promote' but not every test crossed its threshold")
    return problems


def verify_chain(certificates: Sequence[dict]) -> list:
    """Verify a gate's whole history: each certificate, and that budget was
    only ever inherited from the candidate directly before, when that one
    was promoted, and that no candidate number was used twice."""
    problems: list = []
    by_k: dict = {}
    seats: set = set()
    for certificate in certificates:
        if certificate.get("reuse"):
            seat = (certificate.get("batch"), certificate.get("position"))
            if seat in seats:
                problems.append(f"batch {seat[0]} position {seat[1]} was used more than once")
            seats.add(seat)
        k = certificate.get("k")
        if k in by_k:
            problems.append(f"candidate number {k} appears more than once")
        by_k[k] = certificate
        problems += [f"candidate {k}: {problem}" for problem in verify_certificate(certificate)]
    for k, certificate in by_k.items():
        inherited = float(certificate.get("alpha_inherited") or 0.0)
        previous = by_k.get(k - 1) if isinstance(k, int) else None
        allowed = (
            float(previous["alpha"])
            if previous is not None and previous.get("decision") == PROMOTE
            else 0.0
        )
        if inherited > allowed * (1 + 1e-9) + 1e-15:
            problems.append(
                f"candidate {k}: inherited budget {inherited} but at most {allowed} was available"
            )
    return problems


def _check_score(value: float) -> None:
    if not (isinstance(value, (int, float)) and 0.0 <= value <= 1.0):
        raise ValueError(f"scores must be numbers in [0, 1], got {value!r}")
