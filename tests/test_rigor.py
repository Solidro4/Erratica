"""Tests for the pieces that make certificates checkable and durable."""

from __future__ import annotations

import copy
import json
import math
import random
import unittest

from erratica import Erratica, expected_samples, verify_certificate, verify_chain
from erratica.learning.egate import CONTINUE, PROMOTE, REJECT, BettingEProcess, EGate, alpha_for
from erratica.learning.evalstream import Behaviour, Certifier, DriftMonitor, GeneratedStream

from tests.helpers import EngineTestCase
from tests.test_certified import (
    LetterModel,
    RamblingModel,
    has_resolution,
    is_brief,
    letter_certifier,
    support_certifier,
    support_prompt,
)
from tests.helpers import report_failure


def run_gate(gate: EGate, protected_scores, task_scores=None, rng=None):
    """Feed a test from callables returning (candidate, reference) pairs."""
    test = gate.begin(require_improvement=task_scores is not None)
    while test.decision == CONTINUE:
        for name in test.behaviours:
            if not test.settled(name):
                test.add_protected(name, *protected_scores(name))
        if task_scores is not None and not test.settled(None):
            test.add_task(*task_scores())
    return test


class CertificateVerificationTests(unittest.TestCase):
    def _promoted(self) -> dict:
        rng = random.Random(1)
        gate = EGate(["a", "b"], delta=0.05, margin=0.05)
        test = run_gate(
            gate,
            lambda name: (1.0, 1.0) if rng.random() < 0.9 else (1.0, 0.0),
            lambda: (1.0 if rng.random() < 0.7 else 0.0, 0.0),
        )
        self.assertEqual(test.decision, PROMOTE)
        return test.certificate()

    def test_genuine_certificate_verifies_after_a_json_round_trip(self) -> None:
        certificate = json.loads(json.dumps(self._promoted()))
        self.assertEqual(verify_certificate(certificate), [])

    def test_tampering_is_detected(self) -> None:
        genuine = self._promoted()

        forged_data = copy.deepcopy(genuine)
        forged_data["protected"]["a"]["differences"] = [-1.0] * 40
        forged_data["protected"]["a"]["samples"] = 40
        self.assertTrue(verify_certificate(forged_data))

        forged_budget = copy.deepcopy(genuine)
        forged_budget["alpha"] = 0.5
        self.assertTrue(any("own budget" in p for p in verify_certificate(forged_budget)))

        forged_decision = copy.deepcopy(genuine)
        forged_decision["task"]["differences"] = [0.0] * 5
        forged_decision["task"]["samples"] = 5
        forged_decision["task"]["crossed"] = False
        problems = verify_certificate(forged_decision)
        self.assertTrue(any("not every test crossed" in p for p in problems))

        self.assertTrue(verify_certificate({"k": 1}))

    def test_chain_rejects_budget_that_was_never_available(self) -> None:
        gate = EGate(["a"], delta=0.05, margin=0.05)
        first = run_gate(gate, lambda name: (1.0, 1.0)).certificate()
        second = run_gate(gate, lambda name: (1.0, 1.0)).certificate()
        self.assertGreater(second["alpha_inherited"], 0.0)
        self.assertEqual(verify_chain([first, second]), [])

        # Claiming inheritance from a candidate that was not promoted.
        loser = run_gate(EGate(["a"], max_samples=30, tested=0), lambda name: (0.0, 1.0)).certificate()
        self.assertEqual(loser["decision"], REJECT)
        cheat = copy.deepcopy(second)
        problems = verify_chain([loser, cheat])
        self.assertTrue(any("inherited budget" in p for p in problems))

        self.assertTrue(any("more than once" in p for p in verify_chain([first, first])))


class GuaranteeTests(unittest.TestCase):
    def test_no_false_promotion_beyond_the_budget_on_boundary_nulls(self) -> None:
        # Every candidate sits exactly on the margin, the hardest harmful
        # case. Over a sequence, the chance of ANY promotion is at most delta.
        delta, margin = 0.2, 0.1
        sequences, promoted_sequences = 150, 0
        for seed in range(sequences):
            rng = random.Random(seed)
            gate = EGate(["a"], delta=delta, margin=margin, max_samples=60, futility=None)
            any_promoted = False
            for _ in range(6):
                def scores(name):
                    u = rng.random()
                    # P(-1) = 0.25, P(+1) = 0.15: mean exactly -margin.
                    return (0.0, 1.0) if u < 0.25 else ((1.0, 0.0) if u < 0.40 else (1.0, 1.0))

                any_promoted = any_promoted or run_gate(gate, scores).decision == PROMOTE
            promoted_sequences += any_promoted
        # Allow three standard errors above delta before calling it a failure.
        limit = delta + 3 * math.sqrt(delta * (1 - delta) / sequences)
        self.assertLessEqual(promoted_sequences / sequences, limit)

    def test_early_stop_rarely_costs_a_harmless_candidate(self) -> None:
        # A candidate that is exactly as good as baseline, with noisy
        # disagreement, should be lost to the early stop at most ~futility.
        lost, runs = 0, 200
        for seed in range(runs):
            rng = random.Random(1000 + seed)
            test = EGate(["a"], margin=0.1, max_samples=150, futility=0.02).begin(require_improvement=False)
            while test.decision == CONTINUE:
                u = rng.random()
                test.add_protected("a", *((0.0, 1.0) if u < 0.1 else ((1.0, 0.0) if u < 0.2 else (1.0, 1.0))))
            lost += bool(test.gave_up())
        self.assertLessEqual(lost / runs, 0.02 + 3 * math.sqrt(0.02 * 0.98 / runs))

    def test_early_stop_catches_harm_inside_the_margin_region_quickly(self) -> None:
        # Worse than baseline by 0.3: far beyond the margin. Should stop long
        # before max_samples.
        rng = random.Random(7)
        test = EGate(["a"], margin=0.1, max_samples=5000).begin(require_improvement=False)
        while test.decision == CONTINUE:
            test.add_protected("a", *((0.0, 1.0) if rng.random() < 0.3 else (1.0, 1.0)))
        self.assertEqual(test.decision, REJECT)
        self.assertEqual(test.gave_up(), ["a"])
        self.assertLess(test.certificate()["protected"]["a"]["samples"], 80)

    def test_inputs_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            EGate(["a", "a"])
        with self.assertRaises(ValueError):
            EGate(["a"], futility=1.5)
        with self.assertRaises(ValueError):
            EGate(["a"], max_samples=0)
        test = EGate(["a"]).begin()
        for bad in (float("nan"), "1", None, -0.1):
            with self.assertRaises(ValueError):
                test.add_protected("a", bad, 1.0)
        process = BettingEProcess()
        with self.assertRaises(ValueError):
            process.update(float("nan"))


class PlannerTests(unittest.TestCase):
    def test_quiet_behaviour_cost_matches_the_closed_form(self) -> None:
        alpha = alpha_for(1, 0.05)
        plan = expected_samples(alpha, 0.05)
        self.assertAlmostEqual(plan["floor"], -math.log(alpha) / -math.log(0.95))
        # Our bets are capped at 90% of the largest valid bet, so we sit
        # within about 12% of what any valid test could achieve.
        self.assertLess(plan["typical"] / plan["floor"], 1.12)
        self.assertGreaterEqual(plan["typical"], plan["floor"])

    def test_planner_agrees_with_what_the_gate_actually_spends(self) -> None:
        alpha, margin = alpha_for(3, 0.05), 0.1
        plan = expected_samples(alpha, margin, gains=0.1, losses=0.1)
        spent = []
        for seed in range(40):
            rng = random.Random(seed)
            process = BettingEProcess(lower=-1.0 + margin, upper=1.0 + margin)
            while not process.crossed(alpha) and process.n < 5000:
                u = rng.random()
                process.update((1.0 if u < 0.1 else (-1.0 if u < 0.2 else 0.0)) + margin)
            spent.append(process.n)
        average = sum(spent) / len(spent)
        # The real bettor pays a learning overhead on top of the ideal rate.
        self.assertGreater(average, 0.8 * plan["typical"])
        self.assertLess(average, 2.2 * plan["typical"])

    def test_noise_and_harm_make_it_dearer_or_impossible(self) -> None:
        quiet = expected_samples(0.01, 0.1)["typical"]
        noisy = expected_samples(0.01, 0.1, gains=0.15, losses=0.15)["typical"]
        self.assertGreater(noisy, 2 * quiet)
        self.assertEqual(expected_samples(0.01, 0.1, losses=0.2)["typical"], math.inf)
        self.assertIsNone(expected_samples(0.01, 0.1, gains=0.1)["floor"])
        with self.assertRaises(ValueError):
            expected_samples(0.01, 0.1, gains=0.7, losses=0.7)


class DriftMonitorTests(unittest.TestCase):
    def test_no_alarm_while_within_the_margin(self) -> None:
        rng = random.Random(3)
        monitor = DriftMonitor(["a", "b"], margin=0.1, alarm=0.01)
        for _ in range(600):
            for name in ("a", "b"):
                deployed = 1.0 if rng.random() < 0.82 else 0.0   # true drop 0.03
                baseline = 1.0 if rng.random() < 0.85 else 0.0
                monitor.add(name, deployed, baseline)
        self.assertEqual(monitor.alarms(), [])

    def test_alarm_when_a_behaviour_degrades_and_state_survives(self) -> None:
        rng = random.Random(4)
        monitor = DriftMonitor(["a", "b"], margin=0.1, alarm=0.01)
        for _ in range(200):
            monitor.add("a", 1.0, 1.0)
            monitor.add("b", 0.0 if rng.random() < 0.4 else 1.0, 1.0)   # true drop 0.4
        self.assertEqual(monitor.alarms(), ["b"])
        report = monitor.report()
        self.assertLess(report["b"]["mean_change"], -0.3)
        self.assertFalse(report["a"]["alarm"])

        restored = DriftMonitor(["a", "b"], margin=0.1, alarm=0.01)
        restored.restore(json.loads(json.dumps(monitor.state())))
        self.assertEqual(restored.alarms(), ["b"])
        self.assertEqual(restored.report()["b"]["samples"], 200)


class SwappedModel:
    """Behaves like the simulator until someone 'updates' it."""

    def __init__(self) -> None:
        from erratica import RuleAwareDemoModel

        self.inner = RuleAwareDemoModel()
        self.broken = False

    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        answer = self.inner.generate(prompt, system_prompt, route)
        if self.broken and "Learned rules:" in system_prompt:
            answer += " Also," + " and another thing" * 60
        return answer


class EngineRigorTests(EngineTestCase):
    def test_stored_certificates_verify_and_tampering_shows(self) -> None:
        ai = self.engine(model=LetterModel(), enable_recall=False, certifier=letter_certifier())
        for index in range(3):
            report_failure(ai, f"alpha {index}", route="general", issue_type="need_a", required_terms=["A"])
            report_failure(ai, f"beta {index}", route="general", issue_type="need_b", required_terms=["B"])
        first = ai.propose_rule("use A", route="general", issue_type="need_a")
        second = ai.propose_rule("use B", route="general", issue_type="need_b")
        self.assertTrue(first.accepted and second.accepted)

        self.assertEqual(ai.verify_certificates(), [])

        patch = ai.db.get_patch(second.patch_id)
        forged = patch["details"]["certificate"]
        forged["task"]["differences"] = [0.0] * len(forged["task"]["differences"])
        ai.db.update_patch_status(patch_id=second.patch_id, status="promoted", details={"certificate": forged})
        self.assertTrue(ai.verify_certificates())

    def test_rejected_certificates_are_stored_and_verify_too(self) -> None:
        ai = self.engine(model=RamblingModel(), enable_recall=False, certifier=support_certifier(120))
        self.seed_support_failures(ai)
        decisions = ai.run_learning_cycle()
        self.assertTrue(decisions and not any(d.accepted for d in decisions))
        self.assertEqual(ai.verify_certificates(), [])

    def test_watch_raises_an_alarm_when_the_deployed_system_degrades(self) -> None:
        model = SwappedModel()
        certifier = support_certifier()
        ai = self.engine(model=model, enable_recall=False, certifier=certifier)

        # Nothing learned yet: deployed == baseline, nothing to compare.
        idle = ai.watch(items=10)
        self.assertEqual(idle["alarms"], [])
        self.assertEqual(idle["behaviours"]["brevity"]["samples"], 0)

        self.seed_support_failures(ai)
        self.assertTrue(ai.run_learning_cycle()[0].accepted)
        healthy = ai.watch(items=40)
        self.assertEqual(healthy["alarms"], [])
        self.assertEqual(healthy["behaviours"]["brevity"]["samples"], 40)

        model.broken = True   # the certificate no longer describes this system
        degraded = ai.watch(items=60)
        self.assertEqual(degraded["alarms"], ["brevity"])

        # The monitor's evidence is in the database.
        ai.close()
        fresh = support_certifier()
        reopened = Erratica(db_path=self.tmp / "erratica.db", model=model, enable_recall=False, certifier=fresh)
        self.addCleanup(reopened.close)
        self.assertEqual(fresh.monitor.alarms(), ["brevity"])

    def test_watch_needs_certified_mode(self) -> None:
        with self.assertRaises(ValueError):
            self.engine().watch()

    def test_shared_stream_position_is_restored_once(self) -> None:
        shared = GeneratedStream(support_prompt, seed=9)
        certifier = Certifier(
            gate=EGate(["brevity"]),
            protected=[Behaviour("brevity", shared, is_brief, route="support")],
            tasks={"support": Behaviour("resolution", shared, has_resolution)},
        )
        self.assertEqual(certifier.tasks["support"].route, "support")
        for _ in range(7):
            shared.draw()
        state = certifier.state()

        again = GeneratedStream(support_prompt, seed=9)
        restored = Certifier(
            gate=EGate(["brevity"]),
            protected=[Behaviour("brevity", again, is_brief, route="support")],
            tasks={"support": Behaviour("resolution", again, has_resolution, route="support")},
        )
        restored.restore(state)
        self.assertEqual(again.used, 7)

    def test_behaviours_sharing_a_stream_must_share_a_route(self) -> None:
        shared = GeneratedStream(support_prompt)
        with self.assertRaises(ValueError):
            Certifier(
                gate=EGate(["brevity"]),
                protected=[Behaviour("brevity", shared, is_brief, route="finance")],
                tasks={"support": Behaviour("resolution", shared, has_resolution, route="support")},
            )


if __name__ == "__main__":
    unittest.main()
