from __future__ import annotations

import math
import random
import unittest

from erratica.learning.egate import (
    CONTINUE,
    PROMOTE,
    REJECT,
    BettingEProcess,
    EGate,
    alpha_for,
    lower_confidence_bound,
)


class AlphaSpendingTests(unittest.TestCase):
    def test_quadratic_budget_never_exceeds_delta(self) -> None:
        spent = sum(alpha_for(k, 0.05, "quadratic") for k in range(1, 200001))
        self.assertLess(spent, 0.05)
        self.assertGreater(spent, 0.0499)

    def test_log_budget_never_exceeds_delta(self) -> None:
        # Partial sum plus the integral bound on everything after it.
        last = 200000
        spent = sum(alpha_for(k, 0.05) for k in range(1, last + 1))
        tail = 0.05 / (2.11 * math.log(last + 1))
        self.assertLessEqual(spent + tail, 0.05)
        self.assertGreater(spent + tail, 0.0499)

    def test_log_schedule_is_cheaper_for_late_candidates(self) -> None:
        self.assertGreater(alpha_for(5, 0.05), alpha_for(5, 0.05, "quadratic"))
        self.assertGreater(alpha_for(10000, 0.05), 50 * alpha_for(10000, 0.05, "quadratic"))
        need_first = -math.log(alpha_for(1, 0.05))
        need_late = -math.log(alpha_for(10000, 0.05))
        self.assertLess(need_late / need_first, 5)


class BettingEProcessTests(unittest.TestCase):
    def test_leader_starts_cautious_and_follows_the_evidence(self) -> None:
        process = BettingEProcess()
        opening = process.next_bet()
        self.assertEqual(opening, min(process.bets))
        for _ in range(30):
            process.update(0.5)
        self.assertGreater(process.next_bet(), opening)
        self.assertLessEqual(process.next_bet(), max(process.bets))

    def test_never_far_behind_the_best_fixed_bet(self) -> None:
        rng = random.Random(6)
        process = BettingEProcess()
        fixed = [0.0] * len(process.bets)
        for _ in range(500):
            x = rng.choice((-1.0, 0.0, 0.0, 1.0, 1.0))
            process.update(x)
            fixed = [total + math.log1p(bet * x) for total, bet in zip(fixed, process.bets)]
        self.assertGreaterEqual(process.log_wealth, max(fixed) - math.log(2 * len(process.bets)))

    def test_wealth_stays_positive_at_the_worst_observation(self) -> None:
        process = BettingEProcess(lower=-1.0)
        for _ in range(50):
            process.update(1.0)
        process.update(-1.0)
        self.assertTrue(math.isfinite(process.log_wealth))

    def test_null_rarely_crosses(self) -> None:
        # Mean exactly zero is the boundary of H0. Crossing 1/alpha should
        # happen in at most an alpha share of runs (Ville's inequality).
        rng = random.Random(1)
        alpha = 0.05
        crossings = 0
        runs = 400
        for _ in range(runs):
            process = BettingEProcess()
            for _ in range(300):
                process.update(rng.choice((-1.0, 0.0, 0.0, 1.0)))
            crossings += process.crossed(alpha)
        self.assertLessEqual(crossings / runs, alpha)

    def test_real_effect_crosses_quickly(self) -> None:
        rng = random.Random(2)
        process = BettingEProcess()
        steps = 0
        while not process.crossed(0.001) and steps < 2000:
            process.update(1.0 if rng.random() < 0.3 else 0.0)
            steps += 1
        self.assertLess(steps, 200)

    def test_matches_the_vectorised_copy_in_the_stress_script(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "experiments" / "egate_stress.py"
        spec = importlib.util.spec_from_file_location("egate_stress", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        rng = random.Random(3)
        xs = [rng.choice((-0.95, 0.05, 0.05, 1.05)) for _ in range(200)]
        process = BettingEProcess(lower=-0.95, upper=1.05)
        for x in xs:
            process.update(x)
        bettor = module.HedgedBettor(np.array([-0.95]), np.array([1.05]))
        for x in xs:
            vectorised = bettor.update(np.array([x]))
        self.assertAlmostEqual(process.log_wealth, float(vectorised[0]), places=9)


class ConfidenceBoundTests(unittest.TestCase):
    def test_bound_is_below_the_truth_and_tightens_with_data(self) -> None:
        rng = random.Random(8)
        draws = [1.0 if rng.random() < 0.6 else 0.0 for _ in range(2000)]  # true mean 0.6
        few = lower_confidence_bound(draws[:50], 0.05)
        many = lower_confidence_bound(draws, 0.05)
        self.assertLess(few, many)
        self.assertLess(many, 0.6)
        self.assertGreater(many, 0.5)

    def test_bound_covers_the_true_mean_at_the_stated_rate(self) -> None:
        rng = random.Random(9)
        misses = 0
        for _ in range(200):
            draws = [rng.choice((-1.0, 0.0, 1.0)) for _ in range(120)]  # true mean 0
            misses += lower_confidence_bound(draws, 0.1) > 0.0
        self.assertLessEqual(misses / 200, 0.1)

    def test_no_data_gives_the_trivial_bound(self) -> None:
        self.assertEqual(lower_confidence_bound([], 0.05), -1.0)


class EGateTests(unittest.TestCase):
    def test_harmless_improvement_is_promoted_with_a_certificate(self) -> None:
        gate = EGate(["format", "safety"], delta=0.05, margin=0.05)
        test = gate.begin()
        rng = random.Random(4)
        while test.decision == CONTINUE:
            test.add_protected("format", 1.0, 1.0)
            test.add_protected("safety", 1.0, 1.0)
            test.add_task(1.0 if rng.random() < 0.6 else 0.0, 0.0)

        self.assertEqual(test.decision, PROMOTE)
        certificate = test.certificate()
        self.assertEqual(certificate["k"], 1)
        for entry in certificate["protected"].values():
            self.assertGreaterEqual(entry["max_log_e"], certificate["log_threshold"])
            # Unchanged behaviour costs about log(1/alpha) / margin samples.
            self.assertLess(entry["samples"], 2 * certificate["log_threshold"] / 0.05)
            self.assertAlmostEqual(entry["observed_mean_change"], 0.0)

    def test_regression_on_one_behaviour_is_never_promoted(self) -> None:
        rng = random.Random(5)
        gate = EGate(["format", "safety"], max_samples=600)
        for _ in range(40):
            test = gate.begin()
            while test.decision == CONTINUE:
                test.add_protected("format", 1.0, 1.0)
                # Fails safety 10% of the time where the baseline never did.
                test.add_protected("safety", 0.0 if rng.random() < 0.10 else 1.0, 1.0)
                test.add_task(1.0, 0.0)
            self.assertEqual(test.decision, REJECT)

    def test_no_improvement_on_the_task_is_not_promoted(self) -> None:
        gate = EGate(["format"], max_samples=300)
        test = gate.begin()
        while test.decision == CONTINUE:
            test.add_protected("format", 1.0, 1.0)
            test.add_task(0.5, 0.5)
        self.assertEqual(test.decision, REJECT)

    def test_budget_shrinks_with_each_candidate_and_survives_restart(self) -> None:
        gate = EGate(["format"])
        first = gate.begin()
        second = gate.begin()
        self.assertGreater(first.alpha, second.alpha)
        self.assertEqual(second.inherited, 0.0)

        resumed = EGate(["format"], **gate.state())
        self.assertEqual(resumed.begin().k, 3)

    def test_promoted_candidate_passes_its_level_to_the_next(self) -> None:
        gate = EGate(["format"])
        first = gate.begin(require_improvement=False)
        while first.decision == CONTINUE:
            first.add_protected("format", 1.0, 1.0)
        self.assertEqual(first.decision, PROMOTE)

        second = gate.begin(require_improvement=False)
        self.assertAlmostEqual(second.inherited, first.alpha)
        self.assertAlmostEqual(second.alpha, alpha_for(2, 0.05) + first.alpha)

        # A candidate that is not promoted breaks the chain.
        third = gate.begin(require_improvement=False)
        self.assertEqual(third.inherited, 0.0)

        # The inheritance survives a restart.
        gate2 = EGate(["format"])
        winner = gate2.begin(require_improvement=False)
        while winner.decision == CONTINUE:
            winner.add_protected("format", 1.0, 1.0)
        resumed = EGate(["format"], **gate2.state())
        self.assertAlmostEqual(resumed.begin().inherited, winner.alpha)

        self.assertEqual(EGate(["format"], recycle=False, carry=0.01).begin().inherited, 0.0)

    def test_invalid_inputs_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            EGate(["a"], schedule="nope")
        with self.assertRaises(ValueError):
            EGate(["a"]).begin(behaviours=["b"])
        with self.assertRaises(ValueError):
            EGate(["a"], margin=0.0)
        test = EGate(["a"]).begin()
        with self.assertRaises(ValueError):
            test.add_protected("a", 1.2, 1.0)


if __name__ == "__main__":
    unittest.main()
