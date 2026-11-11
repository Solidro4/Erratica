from __future__ import annotations

import unittest

from erratica import Domain
from erratica.cli.benchmark import render_benchmark
from erratica.cli.demo import render_demo
from erratica.learning.clustering import FailureCluster
from erratica.learning.evaluator import PatchEvaluator
from erratica.learning.replay import ReplayCase
from erratica.learning.splits import split_cases
from erratica.learning.triage import MistakeTriage

from tests.helpers import REFUND_IDEAL, REFUND_PROMPT, VAGUE_ANSWER, EngineTestCase, report_failure


class LearningCycleTests(EngineTestCase):
    def test_rule_is_promoted_on_held_out_cases(self) -> None:
        ai = self.engine()
        cases = self.seed_support_failures(ai)

        decisions = ai.run_learning_cycle()

        self.assertEqual(len(decisions), 1)
        decision = decisions[0]
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.validation, "holdout")
        self.assertEqual(decision.route, "support")
        self.assertGreater(decision.cases_scored, 0)
        self.assertLess(decision.cases_scored, len(cases))
        self.assertGreater(decision.score_after, decision.score_before)

        patch = ai.db.get_patch(decision.patch_id)
        self.assertEqual(patch["details"]["validation"], "holdout")
        self.assertEqual(
            patch["details"]["train_cases"] + decision.cases_scored,
            len(cases),
        )

        improved = ai.generate("Customer says order #70555 never showed up and asks what happens now.")
        self.assertIn("Resolution:", improved.text)
        self.assertIn("order 70555", improved.text)

    def test_small_cluster_is_labelled_in_sample(self) -> None:
        ai = self.engine()
        report_failure(
            ai,
            REFUND_PROMPT,
            issue_type="missing_resolution",
            required_terms=["refund", "order"],
            preferred_format="numbered_list",
            ideal_response=REFUND_IDEAL,
        )

        decisions = ai.run_learning_cycle()

        self.assertEqual(len(decisions), 1)
        self.assertTrue(decisions[0].accepted)
        self.assertEqual(decisions[0].validation, "in_sample")
        self.assertIn("in-sample", decisions[0].reason)
        # The case's own correction must not have been recalled into its replay.
        self.assertLess(decisions[0].score_before, 0.2)

    def test_require_holdout_refuses_in_sample_promotion(self) -> None:
        ai = self.engine(require_holdout=True)
        report_failure(
            ai,
            REFUND_PROMPT,
            issue_type="missing_resolution",
            required_terms=["refund", "order"],
            ideal_response=REFUND_IDEAL,
        )

        self.assertEqual(ai.run_learning_cycle(), [])
        self.assertEqual(ai.dashboard()["metrics"]["active_rules"], 0)

    def test_cycle_is_idempotent_until_new_feedback_or_force(self) -> None:
        ai = self.engine()
        self.seed_support_failures(ai)

        self.assertEqual(len(ai.run_learning_cycle()), 1)
        triage_rows = len(ai.db.list_triage(limit=100))
        self.assertEqual(ai.run_learning_cycle(), [])
        self.assertEqual(len(ai.db.list_triage(limit=100)), triage_rows)

        # Forcing re-examines the cluster; the installed rule is not proposed twice.
        ai.run_learning_cycle(force=True)
        self.assertEqual(ai.dashboard()["metrics"]["active_rules"], 1)
        self.assertEqual(len(ai.db.list_triage(limit=100)), triage_rows + 1)

    def test_cycle_interrupted_by_a_model_error_is_retried(self) -> None:
        from erratica import RuleAwareDemoModel

        class FlakyModel:
            def __init__(self) -> None:
                self.inner = RuleAwareDemoModel()
                self.down = False

            def generate(self, prompt, system_prompt, route):
                if self.down:
                    raise RuntimeError("model server is down")
                return self.inner.generate(prompt, system_prompt, route)

        model = FlakyModel()
        ai = self.engine(model=model)
        self.seed_support_failures(ai)

        model.down = True
        with self.assertRaises(RuntimeError):
            ai.run_learning_cycle()
        self.assertEqual(ai.dashboard()["metrics"]["active_rules"], 0)

        model.down = False
        decisions = ai.run_learning_cycle()
        self.assertEqual(len(decisions), 1)
        self.assertTrue(decisions[0].accepted)

    def test_few_shot_promoted_when_no_rule_applies(self) -> None:
        ai = self.engine(enable_recall=False)
        report_failure(ai, REFUND_PROMPT, issue_type="tone_mismatch", ideal_response=REFUND_IDEAL)
        report_failure(
            ai,
            "Customer was promised a refund on order #28844 ten days ago and has heard nothing since.",
            issue_type="tone_mismatch",
            ideal_response="I reviewed your refund. Resolution: refund approved. Timeline: 2 business days.",
        )

        decisions = ai.run_learning_cycle()

        self.assertEqual([d.patch_type for d in decisions], ["few_shot"])
        self.assertTrue(decisions[0].accepted)
        self.assertEqual(ai.dashboard()["metrics"]["active_examples"], 1)
        improved = ai.generate("Customer asks where the refund for order #555 is.")
        self.assertIn("Resolution:", improved.text)

    def test_quarantined_mistakes_are_skipped_and_audited(self) -> None:
        ai = self.engine()
        report_failure(
            ai,
            "Customer asks about their support ticket status.",
            issue_type="unclear",
            notes="not sure what is wrong",
        )

        self.assertEqual(ai.run_learning_cycle(), [])
        snapshot = ai.dashboard()
        self.assertEqual(snapshot["metrics"]["quarantined_clusters"], 1)
        self.assertEqual(snapshot["recent_triage"][0]["verdict"], "quarantined")
        self.assertEqual(snapshot["recent_triage"][0]["route"], "support")

    def test_recall_helps_similar_prompts_before_any_cycle(self) -> None:
        ai = self.engine()
        first = ai.generate(REFUND_PROMPT)
        self.assertEqual(first.text, VAGUE_ANSWER)
        ai.learn_from_feedback(
            interaction_id=first.interaction_id,
            issue_type="missing_resolution",
            ideal_response=REFUND_IDEAL,
        )

        similar = ai.generate(
            "Customer says order #88221 was returned last week and the refund still has not arrived."
        )
        self.assertIn("Resolution:", similar.text)
        self.assertIn("order 88221", similar.text)

    def test_hand_written_rule_goes_through_the_gate(self) -> None:
        ai = self.engine()
        self.seed_support_failures(ai)

        bad = ai.propose_rule(
            "Respond in a single short sentence, avoid detail, and avoid timelines.",
            route="support",
        )
        self.assertFalse(bad.accepted)
        self.assertEqual(bad.validation, "manual")

        good = ai.propose_rule(
            "Always state the resolution and a timeframe in a numbered list.",
            route="support",
        )
        self.assertTrue(good.accepted)
        self.assertEqual(ai.dashboard()["metrics"]["active_rules"], 1)

    def test_demo_and_benchmark_render(self) -> None:
        demo = render_demo()
        self.assertIn("=== Erratica Demo ===", demo)
        self.assertIn("held-out case(s)", demo)
        self.assertIn("PROMOTED", demo)
        self.assertIn("REJECTED", demo)
        self.assertIn("-> kept", demo)

        benchmark = render_benchmark()
        self.assertIn("=== Erratica Benchmark ===", benchmark)
        self.assertIn("Unseen test prompts", benchmark)
        self.assertIn("Recorded failures (learned patches off -> on)", benchmark)
        self.assertIn("PROMOTED", benchmark)
        self.assertIn("REJECTED", benchmark)


class FeedbackTests(EngineTestCase):
    def test_notes_never_become_required_terms(self) -> None:
        ai = self.engine()
        report_failure(ai, REFUND_PROMPT, issue_type="missing_resolution", notes="too vague, no resolution")

        failure = ai.memory.failed_interactions()[0]
        self.assertEqual(failure["required_terms"], [])
        self.assertEqual(failure["notes"], "too vague, no resolution")

    def test_approval_is_a_win_not_a_failure(self) -> None:
        ai = self.engine()
        result = ai.generate(REFUND_PROMPT)
        ai.approve(result.interaction_id)

        self.assertEqual(ai.memory.failed_interactions(), [])
        wins = ai.db.list_wins()
        self.assertEqual(len(wins), 1)
        self.assertEqual(wins[0]["ideal_response"], result.text)
        self.assertEqual(ai.dashboard()["issue_clusters"], [])
        self.assertEqual(ai.run_learning_cycle(), [])

    def test_feedback_on_unknown_interaction_is_an_error(self) -> None:
        ai = self.engine()
        with self.assertRaises(ValueError):
            ai.learn_from_feedback(interaction_id=999, issue_type="anything")

    def test_positive_rating_in_a_cluster_is_dissent_and_is_not_replayed(self) -> None:
        ai = self.engine()
        first = ai.generate(REFUND_PROMPT)
        ai.learn_from_feedback(interaction_id=first.interaction_id, rating=-1, issue_type="too_vague")
        second = ai.generate(REFUND_PROMPT)
        ai.learn_from_feedback(interaction_id=second.interaction_id, rating=1, issue_type="too_vague")

        cluster = ai.clusterer.cluster(ai.memory.failed_interactions())[0]
        self.assertEqual(cluster.size, 2)
        self.assertEqual(len(cluster.failing_cases), 1)
        self.assertFalse(ai.triage.review(cluster).learnable)


class RoutingTests(EngineTestCase):
    def test_route_can_be_given_explicitly(self) -> None:
        ai = self.engine()
        self.assertEqual(ai.generate("Summarise quarterly revenue by segment.").route, "finance")
        self.assertEqual(ai.generate("My refund for the order is late.").route, "support")
        self.assertEqual(ai.generate("Tell me a joke.").route, "general")
        self.assertEqual(ai.generate("Tell me a joke.", route="support").route, "support")

    def test_learned_rules_stay_on_their_route(self) -> None:
        ai = self.engine()
        self.seed_support_failures(ai)
        ai.run_learning_cycle()

        self.assertIn("Learned rules:", ai.policy.render_system_prompt(route="support"))
        self.assertNotIn("Learned rules:", ai.policy.render_system_prompt(route="finance"))
        self.assertNotIn("Learned rules:", ai.policy.render_system_prompt(route="general"))

    def test_custom_domain_brings_its_own_template_and_check(self) -> None:
        def has_citation(case, response):
            return 1.0 if "[source]" in response else 0.0

        research = Domain(
            name="research",
            keywords=("paper", "study", "citation"),
            rule_templates={"missing_citation": "Always cite with [source] markers."},
            issue_checks={"missing_citation": has_citation},
        )

        class CitingModel:
            def generate(self, prompt, system_prompt, route):
                return "Finding. [source]" if "[source]" in system_prompt else "Finding."

        ai = self.engine(model=CitingModel(), domains=[research])
        for topic in ("sleep", "diet", "exercise", "caffeine"):
            report_failure(ai, f"Summarise the study on {topic}.", issue_type="missing_citation")

        decisions = ai.run_learning_cycle()

        self.assertEqual(len(decisions), 1)
        self.assertTrue(decisions[0].accepted)
        self.assertEqual(decisions[0].route, "research")
        self.assertEqual(decisions[0].validation, "holdout")
        self.assertEqual(ai.generate("What does the paper say?").text, "Finding. [source]")


class EvaluatorTests(unittest.TestCase):
    def _case(self, **overrides) -> ReplayCase:
        values = dict(
            interaction_id=1,
            prompt=REFUND_PROMPT,
            baseline_response=VAGUE_ANSWER,
            route="support",
            issue_type="missing_resolution",
            required_terms=["refund", "order"],
            preferred_format="numbered_list",
            ideal_response=REFUND_IDEAL,
        )
        values.update(overrides)
        return ReplayCase(**values)

    def test_better_answer_scores_higher(self) -> None:
        evaluator = PatchEvaluator()
        case = self._case()
        self.assertLess(evaluator.score_case(case, VAGUE_ANSWER), 0.2)
        self.assertGreater(evaluator.score_case(case, REFUND_IDEAL), 0.9)

    def test_custom_similarity_scorer_and_critic_are_used(self) -> None:
        class FixedScorer:
            name = "fixed"

            def score(self, reference, candidate):
                return 1.0

        class HarshCritic:
            name = "harsh"

            def score(self, *, prompt, response, reference=None, route="general"):
                return 0.0

        case = self._case(route="general", issue_type="x", required_terms=[], preferred_format=None)
        self.assertEqual(PatchEvaluator(similarity_scorer=FixedScorer()).score_case(case, "anything"), 1.0)
        self.assertEqual(
            PatchEvaluator(similarity_scorer=FixedScorer(), critics=[HarshCritic()]).score_case(case, "anything"),
            0.5,
        )

    def test_case_without_any_signal_is_unscorable_not_zero(self) -> None:
        evaluator = PatchEvaluator()
        case = self._case(
            route="general",
            issue_type="vibes",
            required_terms=[],
            preferred_format=None,
            ideal_response=None,
        )
        self.assertIsNone(evaluator.score_case(case, "anything"))
        summary = evaluator.evaluate(
            [case],
            [{"interaction_id": 1, "response": "a"}],
            [{"interaction_id": 1, "response": "b"}],
        )
        self.assertEqual(summary.scored, 0)
        self.assertEqual(summary.unscorable, 1)


class SplitTests(unittest.TestCase):
    def test_same_prompt_never_lands_on_both_sides(self) -> None:
        cases = [{"prompt": f"prompt {index % 5}", "n": index} for index in range(20)]
        split = split_cases(cases)

        self.assertTrue(split.is_holdout)
        train_prompts = {case["prompt"] for case in split.train}
        holdout_prompts = {case["prompt"] for case in split.holdout}
        self.assertFalse(train_prompts & holdout_prompts)
        self.assertEqual(len(split.train) + len(split.holdout), 20)
        self.assertTrue(train_prompts and holdout_prompts)

    def test_split_is_deterministic(self) -> None:
        cases = [{"prompt": f"prompt {index}"} for index in range(9)]
        first = split_cases(cases)
        second = split_cases(list(reversed(cases)))
        self.assertEqual(
            sorted(case["prompt"] for case in first.holdout),
            sorted(case["prompt"] for case in second.holdout),
        )

    def test_too_few_distinct_prompts_is_in_sample(self) -> None:
        cases = [{"prompt": "a"}, {"prompt": "A "}, {"prompt": "b"}]
        split = split_cases(cases)
        self.assertFalse(split.is_holdout)
        self.assertEqual(len(split.train), 3)
        self.assertEqual(len(split.holdout), 3)


class MistakeTriageTests(unittest.TestCase):
    def _cluster(self, cases: list[dict]) -> FailureCluster:
        return FailureCluster(issue_type="test_issue", cases=cases, required_terms=[], preferred_formats=[])

    def test_recurring_failures_with_reference_answers_are_learnable(self) -> None:
        cases = [
            {"rating": -1, "ideal_response": "good answer", "preferred_format": "numbered_list"}
            for _ in range(3)
        ]
        verdict = MistakeTriage().review(self._cluster(cases))
        self.assertTrue(verdict.learnable)
        self.assertGreaterEqual(verdict.score, 0.45)

    def test_conflicting_format_feedback_is_quarantined(self) -> None:
        cases = [
            {"rating": -1, "ideal_response": "a", "preferred_format": "paragraph"},
            {"rating": -1, "ideal_response": "b", "preferred_format": "numbered_list"},
        ]
        verdict = MistakeTriage().review(self._cluster(cases))
        self.assertFalse(verdict.learnable)
        self.assertIn("conflicting formats", " ".join(verdict.reasons))

    def test_disagreeing_ratings_are_quarantined(self) -> None:
        cases = [{"rating": 1, "ideal_response": "a"}, {"rating": -1, "ideal_response": "b"}]
        self.assertFalse(MistakeTriage().review(self._cluster(cases)).learnable)

    def test_one_off_without_reference_answer_is_quarantined(self) -> None:
        verdict = MistakeTriage().review(self._cluster([{"rating": -1}]))
        self.assertFalse(verdict.learnable)
        self.assertLess(verdict.score, 0.45)

    def test_critic_evidence_counts_for_less_than_human_evidence(self) -> None:
        human = MistakeTriage().review(self._cluster([{"rating": -1, "ideal_response": "a"}]))
        critic = MistakeTriage().review(
            self._cluster([{"rating": -1, "ideal_response": "a", "source": "critic"}])
        )
        self.assertLess(critic.score, human.score)


if __name__ == "__main__":
    unittest.main()
