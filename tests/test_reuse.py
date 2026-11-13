"""Reusing one evaluation batch across rejected candidates (Theorem 7)."""

from __future__ import annotations

import math
import random
import unittest

from erratica import Erratica, RuleAwareDemoModel, verify_certificate
from erratica.learning.egate import (
    CONTINUE,
    PROMOTE,
    EGate,
    alpha_for_position,
    batch_weight,
    position_weight,
)
from erratica.learning.evalstream import Behaviour, BatchCache, Certifier, GeneratedStream, certify

from tests.helpers import EngineTestCase
from tests.test_certified import has_resolution, is_brief, support_prompt


class BudgetTests(unittest.TestCase):
    def test_two_level_budget_sums_to_at_most_delta(self) -> None:
        positions = sum(position_weight(j) for j in range(1, 100001))
        self.assertLessEqual(positions, 1.0)
        self.assertGreater(positions, 0.9999)
        last = 100000
        batches = sum(batch_weight(e) for e in range(1, last + 1)) + 1 / (2.11 * math.log(last + 1))
        self.assertLessEqual(batches, 1.0)
        self.assertAlmostEqual(alpha_for_position(3, 2, 0.05), 0.05 * batch_weight(3) / 6)

    def test_gate_walks_positions_then_batches(self) -> None:
        gate = EGate(["a"], reuse=True, max_samples=200)
        first = gate.begin(require_improvement=False)
        second = gate.begin(require_improvement=False)   # first was not promoted
        self.assertEqual((first.batch, first.position), (1, 1))
        self.assertEqual((second.batch, second.position), (1, 2))
        self.assertAlmostEqual(second.alpha, alpha_for_position(1, 2, 0.05))

        while second.decision == CONTINUE:
            second.add_protected("a", 1.0, 0.0)
        self.assertEqual(second.decision, PROMOTE)
        third = gate.begin(require_improvement=False)
        self.assertEqual((third.batch, third.position), (2, 1))
        # Only the first candidate of a batch inherits a promoted level.
        self.assertAlmostEqual(third.inherited, second.alpha)
        fourth = gate.begin(require_improvement=False)
        self.assertEqual((fourth.batch, fourth.position, fourth.inherited), (2, 2, 0.0))

        gate.retire_batch()
        self.assertEqual(gate.begin(require_improvement=False).batch, 3)

        resumed = EGate(["a"], reuse=True, **gate.state())
        self.assertEqual((resumed.batch, resumed.position), (3, 1))


class SharedBatchGuaranteeTests(unittest.TestCase):
    """Many worst-case candidates judged on ONE batch.

    The baseline's scores on the batch are shared by every candidate, so
    their tests are strongly dependent. The chance that any of them is
    promoted must still be at most the batch's share of delta.
    """

    DELTA, MARGIN, CANDIDATES, SAMPLES = 0.3, 0.1, 8, 60

    def _run(self, seed: int, identical: bool) -> bool:
        rng = random.Random(seed)
        baseline = [1.0 if rng.random() < 0.8 else 0.0 for _ in range(self.SAMPLES)]
        gate = EGate(["a"], delta=self.DELTA, margin=self.MARGIN, max_samples=self.SAMPLES,
                     reuse=True, futility=None)
        shared_flips = [rng.random() for _ in range(self.SAMPLES)]
        for _ in range(self.CANDIDATES):
            test = gate.begin(require_improvement=False)
            index = 0
            while test.decision == CONTINUE:
                u = shared_flips[index] if identical else rng.random()
                # Mean change exactly -margin: lose 0.2 of the passes, gain 0.3 of the fails.
                if baseline[index] == 1.0:
                    candidate = 0.0 if u < 0.2 else 1.0
                else:
                    candidate = 1.0 if u < 0.3 else 0.0
                test.add_protected("a", candidate, baseline[index])
                index += 1
            if test.decision == PROMOTE:
                return True
        return False

    def _check(self, identical: bool) -> None:
        runs = 200
        promoted = sum(self._run(seed, identical) for seed in range(runs))
        bound = self.DELTA * batch_weight(1)
        self.assertLessEqual(promoted / runs, bound + 3 * math.sqrt(bound * (1 - bound) / runs))

    def test_independent_candidates_on_a_shared_batch(self) -> None:
        self._check(identical=False)

    def test_identical_candidates_on_a_shared_batch(self) -> None:
        self._check(identical=True)


class CertifyReuseTests(unittest.TestCase):
    def test_second_candidate_rereads_the_batch_and_reuses_reference_answers(self) -> None:
        stream = GeneratedStream(lambda rng: f"q{rng.random()}")
        tone = Behaviour("tone", stream, lambda item, r: 1.0 if r == "ok" else 0.0)
        gate = EGate(["tone"], reuse=True, max_samples=300)
        cache = BatchCache()
        baseline_calls = []

        def baseline(prompt, route=None):
            baseline_calls.append(prompt)
            return "ok"

        seen = {"rude": [], "polite": []}

        def rude(prompt, route=None):
            seen["rude"].append(prompt)
            return "no" if len(seen["rude"]) % 3 == 0 else "ok"

        def polite(prompt, route=None):
            seen["polite"].append(prompt)
            return "ok"

        first = certify(gate, candidate=rude, baseline=baseline, protected=[tone], cache=cache)
        second = certify(gate, candidate=polite, baseline=baseline, protected=[tone], cache=cache)

        self.assertEqual(first["decision"], "reject")
        self.assertEqual(second["decision"], "promote")
        self.assertEqual((second["batch"], second["position"]), (1, 2))
        # Same items, same order, from the start of the batch.
        self.assertEqual(seen["polite"][: len(seen["rude"])], seen["rude"])
        # The baseline answered each item exactly once across both tests.
        self.assertEqual(len(baseline_calls), len(set(baseline_calls)))
        self.assertEqual(len(baseline_calls), stream.used)
        self.assertEqual(second["fresh_items"], second["items_drawn"] - first["items_drawn"])
        self.assertEqual(second["model_calls"], second["items_drawn"] + second["fresh_items"])
        self.assertEqual(verify_certificate(second), [])

    def test_cache_needs_a_reuse_gate(self) -> None:
        stream = GeneratedStream(lambda rng: "q")
        tone = Behaviour("tone", stream, lambda item, r: 1.0)
        with self.assertRaises(ValueError):
            certify(EGate(["tone"]), candidate=lambda p, r=None: "", baseline=lambda p, r=None: "",
                    protected=[tone], cache=BatchCache())

    def test_tampered_reuse_certificate_is_caught(self) -> None:
        gate = EGate(["a"], reuse=True)
        gate.begin(require_improvement=False)
        test = gate.begin(require_improvement=False)
        while test.decision == CONTINUE:
            test.add_protected("a", 1.0, 1.0)
        certificate = test.certificate()
        self.assertEqual(verify_certificate(certificate), [])
        certificate["position"] = 1          # claim the larger share of the budget
        self.assertTrue(verify_certificate(certificate))


class EssayModel:
    """A rule containing ESSAY makes replies long; otherwise the simulator."""

    def __init__(self) -> None:
        self.inner = RuleAwareDemoModel()

    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        answer = self.inner.generate(prompt, system_prompt, route)
        if "ESSAY" in system_prompt:
            answer += " In addition," + " and furthermore" * 60
        return answer


def reuse_certifier() -> Certifier:
    stream = GeneratedStream(support_prompt, seed=11)
    return Certifier(
        gate=EGate(["brevity"], delta=0.05, margin=0.05, max_samples=400, reuse=True),
        protected=[Behaviour("brevity", stream, is_brief, route="support")],
        tasks={"support": Behaviour("resolution", stream, has_resolution, route="support")},
    )


class EngineReuseTests(EngineTestCase):
    BAD = "State the resolution and a timeframe in a numbered list. ESSAY"
    GOOD = "State the resolution and a timeframe in a numbered list."

    def test_rejection_is_sealed_until_a_promotion_retires_the_batch(self) -> None:
        certifier = reuse_certifier()
        ai = self.engine(model=EssayModel(), enable_recall=False, certifier=certifier)
        self.seed_support_failures(ai)

        bad = ai.propose_rule(self.BAD, route="support")
        self.assertFalse(bad.accepted)
        self.assertIn("sealed", bad.reason)
        self.assertNotIn("brevity", bad.reason)
        self.assertTrue(bad.certificate["sealed"])
        self.assertNotIn("protected", ai.db.get_patch(bad.patch_id)["details"]["certificate"])
        used_by_first = certifier.protected[0].stream.used
        self.assertEqual(len(certifier.cache), used_by_first)

        good = ai.propose_rule(self.GOOD, route="support")
        self.assertTrue(good.accepted, good.reason)
        self.assertEqual((good.certificate["batch"], good.certificate["position"]), (1, 2))
        # It re-read the first test's items instead of drawing new ones for them.
        self.assertEqual(good.certificate["fresh_items"],
                         good.certificate["items_drawn"] - used_by_first)

        # The batch is retired: details are open, and everything verifies.
        opened = ai.db.get_patch(bad.patch_id)["details"]
        self.assertIn("brevity", opened["reason"])
        self.assertIn("protected", opened["certificate"])
        self.assertEqual(len(certifier.cache), 0)
        self.assertEqual(ai.verify_certificates(), [])
        self.assertEqual(certifier.gate.state()["batch"], 2)

    def test_details_can_be_revealed_early_at_the_price_of_the_batch(self) -> None:
        certifier = reuse_certifier()
        ai = self.engine(model=EssayModel(), enable_recall=False, certifier=certifier)
        self.seed_support_failures(ai)
        bad = ai.propose_rule(self.BAD, route="support")
        self.assertEqual(ai.verify_certificates(), [])     # sealed ones are still checked

        self.assertEqual(ai.reveal_certificates(), 1)
        self.assertIn("brevity", ai.db.get_patch(bad.patch_id)["details"]["reason"])
        used = certifier.protected[0].stream.used

        good = ai.propose_rule(self.GOOD, route="support")
        self.assertTrue(good.accepted, good.reason)
        self.assertEqual((good.certificate["batch"], good.certificate["position"]), (2, 1))
        self.assertEqual(good.certificate["fresh_items"], good.certificate["items_drawn"])
        self.assertEqual(certifier.protected[0].stream.used, used + good.certificate["items_drawn"])

    def test_restart_retires_a_batch_that_was_in_use(self) -> None:
        certifier = reuse_certifier()
        ai = self.engine(model=EssayModel(), enable_recall=False, certifier=certifier)
        self.seed_support_failures(ai)
        ai.propose_rule(self.BAD, route="support")
        ai.close()

        fresh = reuse_certifier()
        reopened = Erratica(db_path=self.tmp / "erratica.db", model=EssayModel(),
                            enable_recall=False, certifier=fresh)
        self.addCleanup(reopened.close)
        self.assertEqual((fresh.gate.batch, fresh.gate.position), (2, 0))
        good = reopened.propose_rule(self.GOOD, route="support")
        self.assertTrue(good.accepted, good.reason)
        self.assertEqual(good.certificate["fresh_items"], good.certificate["items_drawn"])
        self.assertEqual(reopened.verify_certificates(), [])


if __name__ == "__main__":
    unittest.main()
