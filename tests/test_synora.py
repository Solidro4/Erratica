from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from synora.cli.demo import render_demo
from synora.engine.runner import Synora, RuleAwareDemoModel


class ReplayLoopTests(unittest.TestCase):
    def test_learning_cycle_promotes_support_resolution_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "synora.db"
            ai = Synora(db_path=db_path, model=RuleAwareDemoModel())
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
            db_path = Path(tempdir) / "synora.db"
            ai = Synora(db_path=db_path, model=RuleAwareDemoModel())
            try:
                prompt = "Customer says order #55210 was delivered damaged and wants a replacement."
                result = ai.generate(prompt)
                ai.learn_from_feedback(
                    interaction_id=result.interaction_id,
                    rating=-1,
                    issue_type="missing_resolution",
                    required_terms=["order", "replacement"],
                    preferred_format="numbered_list",
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
        self.assertIn("=== Synora Demo ===", output)
        self.assertIn("Replaying 5 failures...", output)
        self.assertIn("Status: PROMOTED", output)
        self.assertIn("Resolution:", output)


if __name__ == "__main__":
    unittest.main()
