from __future__ import annotations

import json
import unittest

from erratica import ModelCritic
from erratica.learning.critic import parse_score

from tests.helpers import REFUND_IDEAL, REFUND_PROMPT, VAGUE_ANSWER, EngineTestCase


class ParseScoreTests(unittest.TestCase):
    def test_accepted_formats(self) -> None:
        self.assertEqual(parse_score("SCORE: 7"), 0.7)
        self.assertEqual(parse_score("I think this is weak.\nScore = 9.5/10"), 0.95)
        self.assertEqual(parse_score("8/10"), 0.8)
        self.assertEqual(parse_score("<think>SCORE: 1</think>\nSCORE: 9"), 0.9)

    def test_unusable_replies_are_none(self) -> None:
        self.assertIsNone(parse_score("looks fine to me"))
        self.assertIsNone(parse_score("SCORE: 11"))
        self.assertIsNone(parse_score(None))

    def test_model_critic_grades_and_revises_through_any_adapter(self) -> None:
        class Grader:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def generate(self, prompt, system_prompt, route):
                self.prompts.append(prompt)
                return "SCORE: 3" if "Grade the answer" in prompt else "  A better answer.  "

        grader = Grader()
        critic = ModelCritic(grader, rubric="Must name a timeframe.")
        self.assertEqual(critic.score(prompt="p", response="r", reference="ref"), 0.3)
        self.assertIn("Must name a timeframe.", grader.prompts[0])
        self.assertIn("ref", grader.prompts[0])
        self.assertEqual(critic.revise(prompt="p", response="r"), "A better answer.")


class FakeCritic:
    """Dislikes the simulator's vague answer and knows a better one."""

    name = "fake_critic"

    def __init__(self, revision: str = REFUND_IDEAL) -> None:
        self.revision = revision

    def score(self, *, prompt, response, reference=None, route="general"):
        return 0.2 if response == VAGUE_ANSWER else 0.9

    def revise(self, *, prompt, response, route="general"):
        return self.revision


class SelfReviewTests(EngineTestCase):
    def test_self_review_flags_weak_answers_and_records_a_correction(self) -> None:
        ai = self.engine()
        weak = ai.generate(REFUND_PROMPT)
        ai.generate("Tell me something.", route="general")

        report = ai.self_review(FakeCritic())

        self.assertEqual((report.reviewed, report.flagged, report.corrections), (2, 1, 1))
        self.assertEqual(report.findings[0].interaction_id, weak.interaction_id)
        failure = ai.memory.failed_interactions()[0]
        self.assertEqual(failure["source"], "critic")
        self.assertEqual(failure["issue_type"], "self_review")
        self.assertEqual(failure["ideal_response"], REFUND_IDEAL)

        # Everything was graded once; a second pass has nothing left to do.
        again = ai.self_review(FakeCritic())
        self.assertEqual(again.reviewed, 0)
        # A critic's approval is not a protected win.
        self.assertEqual(ai.db.list_wins(), [])

    def test_revision_that_is_not_clearly_better_is_not_stored(self) -> None:
        ai = self.engine()
        ai.generate(REFUND_PROMPT)

        report = ai.self_review(FakeCritic(revision=VAGUE_ANSWER))

        self.assertEqual((report.flagged, report.corrections), (1, 0))
        self.assertIsNone(ai.memory.failed_interactions()[0]["ideal_response"])

    def test_critic_corrections_are_not_trusted_by_default(self) -> None:
        ai = self.engine()
        ai.generate(REFUND_PROMPT)
        ai.self_review(FakeCritic())

        # Not recalled at answer time.
        self.assertEqual(ai.generate(REFUND_PROMPT).text, VAGUE_ANSWER)
        # Not exported for fine-tuning.
        self.assertEqual(ai.export_training_data(self.tmp / "a.jsonl").examples_written, 0)
        opted_in = ai.export_training_data(self.tmp / "b.jsonl", include_self_generated=True)
        self.assertEqual(opted_in.examples_written, 1)
        record = json.loads((self.tmp / "b.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(record["messages"][2]["content"], REFUND_IDEAL)

    def test_trust_self_review_opts_in_to_recall(self) -> None:
        ai = self.engine(trust_self_review=True)
        ai.generate(REFUND_PROMPT)
        ai.self_review(FakeCritic())

        self.assertIn("Resolution:", ai.generate(REFUND_PROMPT).text)

    def test_self_review_can_drive_a_validated_patch(self) -> None:
        ai = self.engine()
        prompts = [
            REFUND_PROMPT,
            "Customer says order #55210 arrived damaged and asks for a replacement.",
            "Customer was promised a refund on order #28844 ten days ago and has heard nothing since.",
            "Customer reports the shipment for order #90417 is delayed again and asks for an update.",
        ]
        for prompt in prompts:
            ai.generate(prompt)
        ai.self_review(FakeCritic())

        decisions = ai.run_learning_cycle()

        self.assertEqual([d.patch_type for d in decisions], ["few_shot"])
        self.assertTrue(decisions[0].accepted)
        self.assertEqual(decisions[0].validation, "holdout")
        details = ai.db.get_patch(decisions[0].patch_id)["details"]
        self.assertEqual(details["human_evidence_fraction"], 0.0)
        self.assertIn("from self-review", ai.model_card())


if __name__ == "__main__":
    unittest.main()
