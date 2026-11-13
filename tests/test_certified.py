from __future__ import annotations

import unittest

from erratica import Erratica, RuleAwareDemoModel
from erratica.learning.egate import EGate
from erratica.learning.evalstream import Behaviour, Certifier, GeneratedStream
from erratica.learning.promoter import paired_p_value

from tests.helpers import EngineTestCase, report_failure


def support_prompt(rng) -> str:
    issue = rng.choice(["arrived damaged", "never arrived", "was returned and the refund is missing"])
    return f"Customer says order #{rng.randint(10000, 99999)} {issue}."


def has_resolution(item, response) -> float:
    return 1.0 if "Resolution:" in response else 0.0


def is_brief(item, response) -> float:
    return 1.0 if len(response.split()) <= 80 else 0.0


def support_certifier(max_samples: int = 400) -> Certifier:
    return Certifier(
        gate=EGate(["brevity"], delta=0.05, margin=0.05, max_samples=max_samples),
        protected=[Behaviour("brevity", GeneratedStream(support_prompt, seed=1), is_brief, route="support")],
        tasks={"support": Behaviour("resolution", GeneratedStream(support_prompt, seed=2), has_resolution, route="support")},
    )


class RamblingModel:
    """Learns the rule, and with it starts writing essays."""

    def __init__(self) -> None:
        self.inner = RuleAwareDemoModel()

    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        answer = self.inner.generate(prompt, system_prompt, route)
        if "Resolution:" in answer:
            answer += " Furthermore," + " and so on" * 60
        return answer


class LetterModel:
    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        if "alpha" in prompt and "use A" in system_prompt:
            return "A"
        if "beta" in prompt and "use B" in system_prompt:
            return "B"
        return "?"


def letter_certifier() -> Certifier:
    def make(rng) -> str:
        return f"{rng.choice(['alpha', 'beta'])} {rng.random()}"

    def correct(item, response) -> float:
        return 1.0 if response == ("A" if item.startswith("alpha") else "B") else 0.0

    return Certifier(
        gate=EGate([], delta=0.05),
        tasks={"general": Behaviour("letters", GeneratedStream(make), correct, route="general")},
    )


class CertifiedModeTests(EngineTestCase):
    def test_patch_is_installed_only_with_a_certificate(self) -> None:
        certifier = support_certifier()
        ai = self.engine(enable_recall=False, certifier=certifier)
        self.seed_support_failures(ai)

        decisions = ai.run_learning_cycle()

        self.assertEqual(len(decisions), 1)
        decision = decisions[0]
        self.assertTrue(decision.accepted, decision.reason)
        self.assertIn("certified on fresh samples", decision.reason)
        certificate = decision.certificate
        self.assertEqual(certificate["decision"], "promote")
        self.assertEqual(certificate["k"], 1)
        self.assertTrue(certificate["protected"]["brevity"]["crossed"])
        self.assertTrue(certificate["task"]["crossed"])
        self.assertEqual(certificate["task"]["certified_above"], 0.0)
        self.assertEqual(certificate["protected"]["brevity"]["certified_above"], -0.05)
        self.assertGreater(certificate["task"]["change_at_least"], -0.1)
        self.assertEqual(ai.db.get_patch(decision.patch_id)["details"]["certificate"]["k"], 1)
        self.assertGreater(certifier.protected[0].stream.used, 20)

        # Budget spent and samples used survive a restart.
        used = {name: stream.used for name, stream in certifier._streams().items()}
        ai.close()
        fresh = support_certifier()
        reopened = Erratica(db_path=self.tmp / "erratica.db", enable_recall=False, certifier=fresh)
        self.addCleanup(reopened.close)
        self.assertEqual(fresh.gate.tested, 1)
        self.assertEqual({name: stream.used for name, stream in fresh._streams().items()}, used)
        self.assertGreater(fresh.gate.begin().inherited, 0.0)

    def test_patch_that_passes_the_screen_but_harms_a_protected_behaviour_is_refused(self) -> None:
        certifier = support_certifier(max_samples=120)
        ai = self.engine(model=RamblingModel(), enable_recall=False, certifier=certifier)
        self.seed_support_failures(ai)

        decisions = ai.run_learning_cycle()

        # The held-out feedback cases liked every rung; the fresh samples did not.
        self.assertGreaterEqual(len(decisions), 1)
        for decision in decisions:
            self.assertGreater(decision.delta, 0.3)
            self.assertFalse(decision.accepted)
            self.assertIn("failed certification", decision.reason)
            self.assertIn("evidence that it harms: brevity", decision.reason)
            self.assertLess(decision.certificate["samples_consumed"]["brevity"], 60)
            self.assertIsNone(decision.certificate["protected"]["brevity"]["certified_above"])
        metrics = ai.dashboard()["metrics"]
        self.assertEqual(metrics["active_rules"] + metrics["active_examples"], 0)
        self.assertEqual(certifier.gate.tested, len(decisions))

    def test_candidates_stopped_by_the_screen_spend_no_budget(self) -> None:
        certifier = support_certifier()
        ai = self.engine(enable_recall=False, certifier=certifier)
        self.seed_support_failures(ai)

        decision = ai.propose_rule(
            "Respond in a single short sentence, avoid detail, and avoid timelines.",
            route="support",
        )

        self.assertFalse(decision.accepted)
        self.assertIsNone(decision.certificate)
        self.assertEqual(certifier.gate.tested, 0)
        self.assertEqual(certifier.protected[0].stream.used, 0)

    def test_route_without_an_evaluation_stream_cannot_be_certified(self) -> None:
        ai = self.engine(model=LetterModel(), enable_recall=False, certifier=support_certifier())
        for index in range(3):
            report_failure(ai, f"alpha {index}", route="general", issue_type="need_a", required_terms=["A"])

        decision = ai.propose_rule("use A", route="general")

        self.assertFalse(decision.accepted)
        self.assertIn("no evaluation stream covers route 'general'", decision.reason)

    def test_only_the_newest_patch_of_a_route_can_be_rolled_back(self) -> None:
        certifier = letter_certifier()
        ai = self.engine(model=LetterModel(), enable_recall=False, certifier=certifier)
        for index in range(3):
            report_failure(ai, f"alpha {index}", route="general", issue_type="need_a", required_terms=["A"])
            report_failure(ai, f"beta {index}", route="general", issue_type="need_b", required_terms=["B"])

        first = ai.propose_rule("use A", route="general", issue_type="need_a")
        second = ai.propose_rule("use B", route="general", issue_type="need_b")
        self.assertTrue(first.accepted, first.reason)
        self.assertTrue(second.accepted, second.reason)
        # The second candidate inherited the first one's level.
        self.assertGreater(second.certificate["alpha_inherited"], 0.0)

        with self.assertRaises(ValueError):
            ai.rollback(first.patch_id)
        self.assertTrue(ai.rollback(second.patch_id))
        self.assertTrue(ai.rollback(first.patch_id))

    def test_uncertifiable_operations_are_refused_or_report_only(self) -> None:
        with self.assertRaises(ValueError):
            self.engine(certifier=support_certifier(), db_name="recall.db")

        ai = self.engine(enable_recall=False, certifier=support_certifier())
        self.seed_support_failures(ai)
        ai.run_learning_cycle()
        with self.assertRaises(NotImplementedError):
            ai.run_weight_cycle(lambda path: RuleAwareDemoModel())
        for finding in ai.run_audit():
            self.assertNotEqual(finding.action, "rolled_back")

    def test_certifier_streams_must_match_its_gate(self) -> None:
        with self.assertRaises(ValueError):
            Certifier(gate=EGate(["brevity", "safety"]), protected=[])


class TrialMatchesDeploymentTests(EngineTestCase):
    def test_candidate_example_is_rendered_under_the_installed_limit(self) -> None:
        ai = self.engine(enable_recall=False)
        for index in range(2):
            ai.policy.examples.add_example(route="support", prompt=f"old {index}", response="r")

        trial = ai.policy.render_system_prompt(
            route="support",
            extra_examples=[{"prompt": "candidate", "response": "r"}],
        )
        # Once installed, the candidate would push out the oldest example.
        self.assertIn("candidate", trial)
        self.assertIn("old 1", trial)
        self.assertNotIn("old 0", trial)

        with_recall = ai.policy.render_system_prompt(
            route="support",
            recalled_examples=[{"prompt": "recalled", "response": "r"}],
        )
        self.assertIn("recalled", with_recall)
        self.assertIn("old 0", with_recall)

    def test_bare_prompt_has_no_learned_behavior(self) -> None:
        ai = self.engine()
        self.seed_support_failures(ai)
        ai.run_learning_cycle()
        self.assertIn("Learned rules:", ai.policy.render_system_prompt(route="support"))
        self.assertNotIn("Learned rules:", ai.policy.render_system_prompt(route="support", bare=True))


class PairedPValueTests(unittest.TestCase):
    def test_three_cases_cannot_prove_much(self) -> None:
        self.assertEqual(paired_p_value([1.0, 1.0, 1.0]), 0.125)

    def test_more_consistent_cases_give_stronger_evidence(self) -> None:
        self.assertAlmostEqual(paired_p_value([0.5] * 10), 1 / 1024)
        self.assertLess(paired_p_value([0.4] * 30), 0.001)

    def test_no_effect_is_not_significant(self) -> None:
        self.assertGreater(paired_p_value([0.3, -0.3, 0.2, -0.2]), 0.4)
        self.assertEqual(paired_p_value([0.0, 0.0]), 1.0)
        self.assertIsNone(paired_p_value([]))

    def test_decisions_carry_the_p_value(self) -> None:
        import tempfile
        from pathlib import Path

        from erratica.cli.common import load_cases, record_cases

        with tempfile.TemporaryDirectory() as tempdir:
            with Erratica(db_path=Path(tempdir) / "p.db") as ai:
                record_cases(ai, load_cases())
                decision = ai.run_learning_cycle()[0]
                self.assertEqual(decision.cases_scored, 3)
                self.assertEqual(decision.p_value, 0.125)
                self.assertEqual(ai.db.get_patch(decision.patch_id)["details"]["p_value"], 0.125)


if __name__ == "__main__":
    unittest.main()
