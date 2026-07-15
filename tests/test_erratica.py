from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from erratica.cli.benchmark import render_benchmark
from erratica.cli.demo import render_demo
from erratica.engine.runner import Erratica, RuleAwareDemoModel
from erratica.learning.clustering import FailureCluster
from erratica.learning.evaluator import PatchEvaluator
from erratica.learning.replay import ReplayCase
from erratica.learning.triage import MistakeTriage


class FixedSimilarityScorer:
    name = "fixed_test"

    def score(self, reference: str, candidate: str) -> float:
        if "approving the refund" in candidate:
            return 0.95
        return 0.05


class ReplayLoopTests(unittest.TestCase):
    def test_learning_cycle_promotes_support_resolution_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "erratica.db"
            ai = Erratica(db_path=db_path, model=RuleAwareDemoModel())
            try:
                prompt = "Customer says order #41327 was returned two weeks ago and the refund still has not arrived."

                first = ai.generate(prompt)
                self.assertNotIn("Resolution:", first.text)

                ai.learn_from_feedback(
                    interaction_id=first.interaction_id,
                    rating=-1,
                    issue_type="missing_resolution",
                    required_terms=["refund", "order"],
                    preferred_format="numbered_list",
                    ideal_response="I reviewed your request about your refund request. 1. Reference: order 41327 2. Resolution: we are approving the refund back to the original payment method 3. Timeline: 3-5 business days 4. Next step: we will send a confirmation update as soon as the action is complete.",
                    notes="too vague, no resolution",
                )

                decisions = ai.run_learning_cycle()
                self.assertEqual(len(decisions), 1)
                self.assertTrue(decisions[0].accepted)
                self.assertGreater(decisions[0].score_after, decisions[0].score_before)

                second = ai.generate(prompt)
                self.assertIn("Resolution:", second.text)
                self.assertIn("refund", second.text.lower())
                self.assertIn("order 41327", second.text.lower())
                self.assertIn("1.", second.text)
            finally:
                ai.close()

    def test_dashboard_tracks_promoted_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "erratica.db"
            ai = Erratica(db_path=db_path, model=RuleAwareDemoModel())
            try:
                prompt = "Customer says order #55210 was delivered damaged and wants a replacement."
                result = ai.generate(prompt)
                ai.learn_from_feedback(
                    interaction_id=result.interaction_id,
                    rating=-1,
                    issue_type="missing_resolution",
                    required_terms=["order", "replacement"],
                    preferred_format="numbered_list",
                    ideal_response="I reviewed your request about a damaged item delivery. 1. Reference: order 55210 2. Resolution: we are sending a replacement shipment at no extra cost 3. Timeline: 1 business day 4. Next step: we will send a confirmation update as soon as the action is complete.",
                    notes="too vague, no resolution",
                )
                ai.run_learning_cycle()

                snapshot = ai.dashboard()
                self.assertEqual(snapshot["metrics"]["promoted_patches"], 1)
                self.assertEqual(snapshot["metrics"]["active_rules"], 1)
                self.assertEqual(snapshot["issue_clusters"][0]["issue_type"], "missing_resolution")
            finally:
                ai.close()

    def test_demo_render_shows_promoted_patch(self) -> None:
        dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "support_replay_cases.json"
        output = render_demo(dataset_path)
        self.assertIn("=== Erratica Demo ===", output)
        self.assertIn("Replaying 5 failures...", output)
        self.assertIn("Status: PROMOTED", output)
        self.assertIn("Ideal response:", output)
        self.assertIn("Resolution:", output)

    def test_evaluator_accepts_custom_similarity_scorer(self) -> None:
        evaluator = PatchEvaluator(similarity_scorer=FixedSimilarityScorer())
        case = ReplayCase(
            interaction_id=1,
            prompt="Customer says order #41327 was returned two weeks ago and the refund still has not arrived.",
            baseline_response="Thanks for reaching out. We are reviewing the issue and will follow up soon.",
            route="support",
            issue_type="missing_resolution",
            required_terms=["refund", "order"],
            preferred_format="numbered_list",
            ideal_response="I reviewed your request about your refund request. 1. Reference: order 41327 2. Resolution: we are approving the refund back to the original payment method 3. Timeline: 3-5 business days 4. Next step: we will send a confirmation update as soon as the action is complete.",
            correction=None,
            notes="too vague, no resolution",
        )
        summary = evaluator.evaluate(
            [case],
            [{"interaction_id": 1, "route": "support", "response": case.baseline_response}],
            [{"interaction_id": 1, "route": "support", "response": "I reviewed your request about your refund request.\n1. Reference: order 41327\n2. Resolution: we are approving the refund back to the original payment method\n3. Timeline: 3-5 business days\n4. Next step: we will send a confirmation update as soon as the action is complete."}],
        )
        self.assertGreater(summary.score_after, summary.score_before)

    def test_benchmark_render_shows_promoted_and_rejected_updates(self) -> None:
        dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "support_replay_cases.json"
        output = render_benchmark(dataset_path)
        self.assertIn("=== Erratica Benchmark ===", output)
        self.assertIn("Case improvements:", output)
        self.assertIn("Behavior update outcomes:", output)
        self.assertIn("REJECTED", output)
        self.assertIn("PROMOTED", output)


class MistakeTriageTests(unittest.TestCase):
    def _cluster(self, cases: list[dict], issue_type: str = "test_issue") -> FailureCluster:
        return FailureCluster(
            issue_type=issue_type,
            cases=cases,
            required_terms=[],
            preferred_formats=[],
        )

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
        cases = [
            {"rating": 1, "ideal_response": "a"},
            {"rating": -1, "ideal_response": "b"},
        ]
        verdict = MistakeTriage().review(self._cluster(cases))
        self.assertFalse(verdict.learnable)

    def test_one_off_without_reference_answer_is_quarantined(self) -> None:
        verdict = MistakeTriage().review(self._cluster([{"rating": -1}]))
        self.assertFalse(verdict.learnable)
        self.assertLess(verdict.score, 0.45)


class _IdealTunedModel:
    """Stands in for a model whose weights were fine-tuned on the exported
    dataset: it answers known prompts with the trained ideal response."""

    def __init__(self, dataset_path: Path) -> None:
        self.answers: dict[str, str] = {}
        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            messages = {message["role"]: message["content"] for message in record["messages"]}
            self.answers[messages["user"]] = messages["assistant"]

    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        return self.answers.get(prompt, "Received.")


class _DegradedModel:
    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        return "ok"


class WeightLoopTests(unittest.TestCase):
    _PROMPT = "Customer says order #41327 was returned two weeks ago and the refund still has not arrived."
    _IDEAL = (
        "I reviewed your request about your refund request.\n"
        "1. Reference: order 41327\n"
        "2. Resolution: we are approving the refund back to the original payment method\n"
        "3. Timeline: 3-5 business days\n"
        "4. Next step: we will send a confirmation update as soon as the action is complete."
    )

    def _seed(self, ai: Erratica) -> None:
        result = ai.generate(self._PROMPT)
        ai.learn_from_feedback(
            interaction_id=result.interaction_id,
            rating=-1,
            issue_type="missing_resolution",
            required_terms=["refund", "order"],
            preferred_format="numbered_list",
            ideal_response=self._IDEAL,
            notes="too vague",
        )

    def test_winning_fine_tune_is_promoted_and_swapped_in(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = Erratica(db_path=Path(tempdir) / "erratica.db", model=RuleAwareDemoModel())
            try:
                self._seed(ai)
                original_model = ai.model
                report = ai.run_weight_cycle(
                    lambda dataset_path: _IdealTunedModel(dataset_path),
                    dataset_path=Path(tempdir) / "sft.jsonl",
                )
                self.assertTrue(report.accepted)
                self.assertGreater(report.score_after, report.score_before)
                self.assertIsNot(ai.model, original_model)
                self.assertIsInstance(ai.model, _IdealTunedModel)

                card = ai.model_card(model_name="tuned")
                self.assertIn("model_weights", card)
                self.assertIn("Fine-tuned weights on", card)
            finally:
                ai.close()

    def test_degraded_fine_tune_is_rejected_and_model_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = Erratica(db_path=Path(tempdir) / "erratica.db", model=RuleAwareDemoModel())
            try:
                self._seed(ai)
                ai.run_learning_cycle()
                original_model = ai.model
                report = ai.run_weight_cycle(
                    lambda dataset_path: _DegradedModel(),
                    dataset_path=Path(tempdir) / "sft.jsonl",
                )
                self.assertFalse(report.accepted)
                self.assertIs(ai.model, original_model)
            finally:
                ai.close()

    def test_weight_cycle_without_experience_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = Erratica(db_path=Path(tempdir) / "erratica.db", model=RuleAwareDemoModel())
            try:
                original_model = ai.model
                report = ai.run_weight_cycle(
                    lambda dataset_path: _DegradedModel(),
                    dataset_path=Path(tempdir) / "sft.jsonl",
                )
                self.assertFalse(report.accepted)
                self.assertIs(ai.model, original_model)
            finally:
                ai.close()


class LearningUpgradeTests(unittest.TestCase):
    _REFUND_PROMPT = "Customer says order #41327 was returned two weeks ago and the refund still has not arrived."
    _REFUND_IDEAL = (
        "I reviewed your request about your refund request.\n"
        "1. Reference: order 41327\n"
        "2. Resolution: we are approving the refund back to the original payment method\n"
        "3. Timeline: 3-5 business days\n"
        "4. Next step: we will send a confirmation update as soon as the action is complete."
    )

    def test_few_shot_example_promoted_when_rule_patch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = Erratica(db_path=Path(tempdir) / "erratica.db", model=RuleAwareDemoModel())
            try:
                result = ai.generate(self._REFUND_PROMPT)
                ai.learn_from_feedback(
                    interaction_id=result.interaction_id,
                    rating=-1,
                    issue_type="tone_mismatch",
                    ideal_response=self._REFUND_IDEAL,
                )

                decisions = ai.run_learning_cycle()
                self.assertEqual(len(decisions), 2)
                self.assertEqual(decisions[0].patch_type, "prompt_rule")
                self.assertFalse(decisions[0].accepted)
                self.assertEqual(decisions[1].patch_type, "few_shot")
                self.assertTrue(decisions[1].accepted)

                snapshot = ai.dashboard()
                self.assertEqual(snapshot["metrics"]["active_examples"], 1)

                improved = ai.generate(self._REFUND_PROMPT)
                self.assertIn("Resolution:", improved.text)
            finally:
                ai.close()

    def test_experience_recall_improves_similar_prompts_without_learning_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = Erratica(db_path=Path(tempdir) / "erratica.db", model=RuleAwareDemoModel())
            try:
                result = ai.generate(self._REFUND_PROMPT)
                self.assertNotIn("Resolution:", result.text)
                ai.learn_from_feedback(
                    interaction_id=result.interaction_id,
                    rating=-1,
                    issue_type="missing_resolution",
                    ideal_response=self._REFUND_IDEAL,
                    notes="too vague",
                )

                similar = ai.generate(
                    "Customer says order #88221 was returned last week and the refund still has not arrived."
                )
                self.assertIn("Resolution:", similar.text)
                self.assertIn("order 88221", similar.text)
            finally:
                ai.close()

    def test_quarantined_mistakes_are_skipped_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = Erratica(db_path=Path(tempdir) / "erratica.db", model=RuleAwareDemoModel())
            try:
                result = ai.generate("Customer asks about their support ticket status.")
                ai.learn_from_feedback(
                    interaction_id=result.interaction_id,
                    rating=-1,
                    issue_type="unclear",
                    notes="not sure what is wrong",
                )

                decisions = ai.run_learning_cycle()
                self.assertEqual(decisions, [])

                snapshot = ai.dashboard()
                self.assertEqual(snapshot["metrics"]["quarantined_clusters"], 1)
                self.assertEqual(snapshot["recent_triage"][0]["verdict"], "quarantined")
            finally:
                ai.close()

    def test_model_card_documents_learning_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = Erratica(db_path=Path(tempdir) / "erratica.db", model=RuleAwareDemoModel())
            try:
                result = ai.generate(self._REFUND_PROMPT)
                ai.learn_from_feedback(
                    interaction_id=result.interaction_id,
                    rating=-1,
                    issue_type="missing_resolution",
                    required_terms=["refund", "order"],
                    preferred_format="numbered_list",
                    ideal_response=self._REFUND_IDEAL,
                    notes="too vague",
                )
                ai.run_learning_cycle()

                card = ai.model_card(model_name="test-model")
                self.assertIn("# Model Card: test-model", card)
                self.assertIn("Behavior updates promoted: 1", card)
                self.assertIn("prompt_rule", card)
                self.assertIn("Validation pass rate: 1/1", card)

                card_path = ai.write_model_card(Path(tempdir) / "MODELCARD.md", model_name="test-model")
                self.assertTrue(card_path.exists())
            finally:
                ai.close()

    def test_export_training_data_writes_chat_format_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = Erratica(db_path=Path(tempdir) / "erratica.db", model=RuleAwareDemoModel())
            try:
                result = ai.generate(self._REFUND_PROMPT)
                ai.learn_from_feedback(
                    interaction_id=result.interaction_id,
                    rating=-1,
                    issue_type="missing_resolution",
                    required_terms=["refund", "order"],
                    preferred_format="numbered_list",
                    ideal_response=self._REFUND_IDEAL,
                    notes="too vague",
                )
                ai.run_learning_cycle()

                export_path = Path(tempdir) / "sft.jsonl"
                report = ai.export_training_data(export_path)
                self.assertGreaterEqual(report.examples_written, 1)
                self.assertTrue(export_path.exists())

                lines = export_path.read_text(encoding="utf-8").strip().splitlines()
                self.assertEqual(len(lines), report.examples_written)
                record = json.loads(lines[0])
                roles = [message["role"] for message in record["messages"]]
                self.assertEqual(roles, ["system", "user", "assistant"])
                self.assertEqual(record["messages"][2]["content"], self._REFUND_IDEAL)
            finally:
                ai.close()


if __name__ == "__main__":
    unittest.main()
