"""Stress test of the e-value promotion gate against the heuristic gate.

Simulates long sequences of candidate patches, most of which are regressions
on one protected behaviour while LOOKING like improvements on the task, and
counts how many each gate lets through and what certification costs.

    python experiments/egate_stress.py --sequences 20 --patches 5000

    # the first version of the gate, for comparison
    python experiments/egate_stress.py --bettor akelly --schedule quadratic --no-recycle

Needs numpy (the library itself does not). ``HedgedBettor`` is a vectorised
copy of ``erratica.learning.egate.BettingEProcess``; ``tests/test_egate.py``
checks the two agree.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

GRID_SIZE = 12
GRID_RATIO = 0.66
PRIOR_WEIGHT = 0.5
LOG_SERIES = 2.11


class HedgedBettor:
    """Grid mixture + follow-the-leader, half and half. State has a trailing grid axis."""

    def __init__(self, lower, upper):
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        self.bets = (0.9 / np.abs(lower))[..., None] * GRID_RATIO ** np.arange(GRID_SIZE)
        self.prior = PRIOR_WEIGHT * (
            np.log1p(self.bets * lower[..., None]) + np.log1p(self.bets * upper[..., None])
        )
        self.fixed = None
        self.leader = None

    def update(self, x):
        x = np.asarray(x, dtype=float)
        if self.fixed is None:
            self.fixed = np.zeros(x.shape + (GRID_SIZE,))
            self.leader = np.zeros(x.shape)
        best = np.argmax(self.fixed + self.prior, axis=-1)
        bets = np.broadcast_to(self.bets, self.fixed.shape)
        chosen = np.take_along_axis(bets, best[..., None], axis=-1)[..., 0]
        self.leader = self.leader + np.log1p(chosen * x)
        self.fixed = self.fixed + np.log1p(bets * x[..., None])
        peak = self.fixed.max(axis=-1)
        mixture = peak + np.log(np.exp(self.fixed - peak[..., None]).mean(axis=-1))
        top = np.maximum(mixture, self.leader)
        return top + np.log(0.5 * np.exp(mixture - top) + 0.5 * np.exp(self.leader - top))


class KellyBettor:
    """The first version: one plug-in bet from the running mean and variance."""

    def __init__(self, lower, upper):
        self.lam_max = 0.9 / np.abs(np.asarray(lower, dtype=float))
        self.n = 0
        self.total = None

    def update(self, x):
        if self.total is None:
            self.total = np.zeros(x.shape)
            self.total_sq = np.zeros(x.shape)
            self.log_wealth = np.zeros(x.shape)
        mean = self.total / (self.n + 1)
        spread = self.total_sq - 2.0 * mean * self.total + self.n * mean * mean
        variance = (0.25 + spread) / (self.n + 1)
        bet = np.minimum(self.lam_max, np.where(mean > 0, mean / (variance + mean * mean), 0.0))
        self.log_wealth = self.log_wealth + np.log1p(bet * x)
        self.total += x
        self.total_sq += x * x
        self.n += 1
        return self.log_wealth


def own_alpha(k, delta, schedule):
    if schedule == "quadratic":
        return 6.0 * delta / (np.pi * k) ** 2
    return delta / (LOG_SERIES * (k + 1) * np.log(k + 1) ** 2)


def draw(rng, mean, noise, size):
    """Paired differences in {-1, 0, +1} with the given mean."""
    up = noise + np.maximum(mean, 0.0)
    down = noise + np.maximum(-mean, 0.0)
    u = rng.random(size)
    return np.where(u < up, 1.0, np.where(u < up + down, -1.0, 0.0))


def costs(samples_used, batch_of, reuse):
    """Fresh evaluation items and model calls. Each item needs one answer
    from the candidate and two reference answers (baseline and incumbent).
    Without reuse every candidate draws its own items and references; with
    reuse a batch's items and references are shared by its candidates."""
    candidate_calls = float(samples_used.sum())
    if reuse:
        longest = np.zeros(batch_of.max() + 1)
        np.maximum.at(longest, batch_of, samples_used)
        fresh = float(longest.sum())
    else:
        fresh = candidate_calls
    return {"fresh_items": fresh, "model_calls": candidate_calls + 2.0 * fresh}


def simulate(seed, args):
    patches, behaviours, margin = args.patches, args.behaviours, args.margin

    def world(rng):
        is_bad = rng.random(patches) < args.bad_share
        means = np.zeros((patches, behaviours))
        victim = rng.integers(0, behaviours, patches)
        boundary = rng.random(patches) < 0.5
        # Half the bad patches sit exactly on the margin: the hardest case for the test.
        harm = np.where(boundary, -margin, -rng.uniform(margin, 0.15, patches))
        means[np.arange(patches), victim] = np.where(is_bad, harm, 0.0)
        # Every patch, good or bad, looks like a real improvement on the task.
        task = np.full((patches, 1), args.task_gain)
        return is_bad, np.concatenate([means, task], axis=1)

    lower = np.array([-1.0 + margin] * behaviours + [-1.0])
    upper = lower + 2.0
    shift = np.array([margin] * behaviours + [0.0])
    make = HedgedBettor if args.bettor == "hedged" else KellyBettor

    stop_threshold = -math.log(args.futility) if args.futility else None

    def run(log_threshold=None):
        """One pass over the evaluation samples. Without thresholds, returns
        the highest evidence each test reached before it was stopped; with
        them, when it crossed. A test is stopped early once the mirror test
        finds evidence that the candidate is worse than baseline at all."""
        rng = np.random.default_rng(seed)
        is_bad, all_means = world(rng)
        bettor = make(lower, upper)
        mirror = HedgedBettor(np.full(behaviours, -1.0), np.full(behaviours, 1.0))
        peak = np.full(all_means.shape, -np.inf)
        crossed_at = np.full(all_means.shape, -1)
        stopped_at = np.full(patches, -1)
        for n in range(args.max_samples):
            raw = draw(rng, all_means, args.noise, all_means.shape)
            log_wealth = bettor.update(raw + shift)
            alive = (stopped_at < 0)[:, None]
            peak = np.where(alive, np.maximum(peak, log_wealth), peak)
            if log_threshold is not None:
                newly = alive & (crossed_at < 0) & (log_wealth >= log_threshold[:, None])
                crossed_at[newly] = n + 1
            if stop_threshold is not None:
                against = mirror.update(-raw[:, :behaviours])
                stop_now = (stopped_at < 0) & (against >= stop_threshold).any(axis=1)
                stopped_at[stop_now] = n + 1
        return rng, is_bad, all_means, peak, crossed_at, stopped_at

    # Pass 1: how much evidence did each test reach?
    _, is_bad, _, peak, _, _ = run()

    # The gate runs candidates in order; a promotion hands its level to the next.
    k = np.arange(1, patches + 1)
    base = own_alpha(k, args.delta, args.schedule)
    level = np.empty(patches)
    promoted = np.zeros(patches, dtype=bool)
    batch_of = np.empty(patches, dtype=int)
    carry = 0.0
    batch, position = 1, 0
    for index in range(patches):
        position += 1
        if args.reuse:
            # Two-level budget: a share per batch, split over its candidates.
            own = (args.delta / (LOG_SERIES * (batch + 1) * math.log(batch + 1) ** 2)
                   / (position * (position + 1)))
            inherited = carry if (args.recycle and position == 1) else 0.0
            base[index] = own
        else:
            own = base[index]
            inherited = carry if args.recycle else 0.0
        level[index] = own + inherited
        promoted[index] = bool((peak[index] >= -math.log(level[index])).all())
        carry = level[index] if promoted[index] else 0.0
        batch_of[index] = batch
        if promoted[index]:
            batch, position = batch + 1, 0

    # Pass 2 (same random draws): when did each test end?
    rng, _, all_means, _, crossed_at, stopped_at = run(-np.log(level))
    samples_used = np.where(
        promoted,
        crossed_at.max(axis=1),
        np.where(stopped_at > 0, stopped_at, args.max_samples),
    )

    # The heuristic gate: mean over 3 held-out cases.
    small = draw(rng, all_means[:, :, None], args.noise, all_means.shape + (3,)).mean(axis=2)
    heuristic = (small[:, -1] > 0.05) & (small[:, :-1].min(axis=1) >= -0.02)

    good = ~is_bad
    early = good & promoted & (k <= 100)
    late = good & promoted & (k > patches - patches // 10)
    return {
        "bad": int(is_bad.sum()),
        "good": int(good.sum()),
        "egate_false": int((promoted & is_bad).sum()),
        "egate_true": int((promoted & good).sum()),
        "heuristic_false": int((heuristic & is_bad).sum()),
        "heuristic_true": int((heuristic & good).sum()),
        "samples_early": float(samples_used[early].mean()) if early.any() else float("nan"),
        "samples_late": float(samples_used[late].mean()) if late.any() else float("nan"),
        "inherited": float((level > base * 1.0000001).mean()),
        "samples_bad": float(samples_used[is_bad & ~promoted].mean()),
        "samples_total": float(samples_used.sum()),
        **costs(samples_used, batch_of, args.reuse),
        "good_stopped_early": int((good & ~promoted & (stopped_at > 0)).sum()),
        "false_at": [int(value) for value in k[promoted & is_bad]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sequences", type=int, default=20)
    parser.add_argument("--patches", type=int, default=5000)
    parser.add_argument("--behaviours", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--bad-share", type=float, default=0.9)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--task-gain", type=float, default=0.10)
    parser.add_argument("--bettor", choices=("hedged", "akelly"), default="hedged")
    parser.add_argument("--schedule", choices=("log", "quadratic"), default="log")
    parser.add_argument("--no-recycle", dest="recycle", action="store_false")
    parser.add_argument("--reuse", action="store_true",
                        help="reuse one evaluation batch until a candidate is promoted")
    parser.add_argument("--futility", type=float, default=0.02,
                        help="early-stop level; 0 disables early stopping")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    totals: dict = {}
    sequences_with_false = {"egate": 0, "heuristic": 0}
    false_at: list = []
    started = time.time()
    for index in range(args.sequences):
        result = simulate(args.seed * 100003 + index, args)
        false_at.extend(result.pop("false_at"))
        for key, value in result.items():
            totals.setdefault(key, []).append(value)
        sequences_with_false["egate"] += result["egate_false"] > 0
        sequences_with_false["heuristic"] += result["heuristic_false"] > 0

    bad = sum(totals["bad"])
    good = sum(totals["good"])
    print(f"{args.sequences} sequences x {args.patches} patches "
          f"({bad} regressions, {good} genuine improvements)")
    print(f"delta={args.delta} margin={args.margin} max_samples={args.max_samples} "
          f"bettor={args.bettor} schedule={args.schedule} recycle={args.recycle} "
          f"futility={args.futility} reuse={args.reuse}")
    print(f"elapsed: {time.time() - started:.0f}s")
    print()
    for name in ("egate", "heuristic"):
        false = sum(totals[f"{name}_false"])
        true = sum(totals[f"{name}_true"])
        print(f"{name:>9}: regressions promoted {false}/{bad} ({false / bad:.4%}); "
              f"good patches promoted {true}/{good} ({true / good:.1%}); "
              f"sequences with at least one regression promoted: "
              f"{sequences_with_false[name]}/{args.sequences}")
    print()
    print(f"e-gate samples to certify a good patch: {np.nanmean(totals['samples_early']):.0f} "
          f"for the first 100 candidates, {np.nanmean(totals['samples_late']):.0f} for the last 10%")
    print(f"candidates tested with inherited budget: {np.mean(totals['inherited']):.1%}")
    print(f"samples spent on a rejected regression: {np.mean(totals['samples_bad']):.0f} on average")
    installed = sum(totals["egate_true"])
    print(f"all-in cost: {sum(totals['samples_total']) / max(1, installed):.0f} samples per installed "
          f"good patch (including everything spent on rejected candidates)")
    print(f"good patches lost to early stopping: {sum(totals['good_stopped_early'])}/{good}")
    print(f"per installed good patch: {sum(totals['fresh_items']) / max(1, installed):.0f} fresh "
          f"evaluation items, {sum(totals['model_calls']) / max(1, installed):.0f} model calls")
    if false_at:
        print(f"position in its sequence of each regression the e-gate promoted: {sorted(false_at)}")


if __name__ == "__main__":
    main()
