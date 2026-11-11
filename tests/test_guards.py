from __future__ import annotations

from tests.helpers import EngineTestCase, report_failure


class TradeoffModel:
    """A rule fixes 'alpha' prompts but wrecks everything else."""

    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        strict = "always cite the ticket" in system_prompt.lower()
        if "alpha" in prompt:
            return "Ticket cited: alpha" if strict else "no idea"
        return "garbage" if strict else "The capital of France is Paris."


class SafeModel(TradeoffModel):
    """Same fix, without the side effect."""

    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        if "alpha" in prompt:
            return super().generate(prompt, system_prompt, route)
        return "The capital of France is Paris."


class RegressionGuardTests(EngineTestCase):
    def _seed(self, ai) -> None:
        for name in ("alpha one", "alpha two", "alpha three"):
            report_failure(ai, name, route="general", issue_type="no_ticket", required_terms=["ticket"])
        good = ai.generate("What is the capital of France?", route="general")
        ai.approve(good.interaction_id)

    def test_rule_that_breaks_an_approved_answer_is_rejected(self) -> None:
        ai = self.engine(model=TradeoffModel())
        self._seed(ai)

        decision = ai.propose_rule("Always cite the ticket.", route="general")

        self.assertFalse(decision.accepted)
        self.assertGreater(decision.delta, 0.5)
        self.assertIn("approved", decision.reason)
        self.assertLess(decision.regression_delta, 0)
        self.assertEqual(ai.dashboard()["metrics"]["active_rules"], 0)

    def test_same_rule_is_promoted_when_approved_answers_survive(self) -> None:
        ai = self.engine(model=SafeModel())
        self._seed(ai)

        decision = ai.propose_rule("Always cite the ticket.", route="general")

        self.assertTrue(decision.accepted)
        self.assertEqual(ai.db.get_patch(decision.patch_id)["details"]["win_cases"], 1)

    def test_learning_cycle_applies_the_same_guard(self) -> None:
        from erratica import Domain

        domain = Domain(
            name="general",
            rule_templates={"no_ticket": "Always cite the ticket."},
        )
        ai = self.engine(model=TradeoffModel(), domains=[domain])
        self._seed(ai)

        decisions = ai.run_learning_cycle()

        self.assertEqual(len(decisions), 1)
        self.assertFalse(decisions[0].accepted)
        self.assertIn("approved", decisions[0].reason)


class AuditTests(EngineTestCase):
    def _install_dead_rule(self, ai, text: str = "Be polite.") -> int:
        patch_id = ai.db.insert_patch(
            patch_type="prompt_rule",
            target="system_prompt",
            content={"rule_text": text},
            rationale="test",
            source_issue_type="test",
            route="support",
        )
        ai.db.update_patch_status(patch_id=patch_id, status="promoted")
        ai.policy.apply_prompt_rule(text, source_patch_id=patch_id, route="support")
        return patch_id

    def test_audit_keeps_what_helps_and_prunes_what_does_not(self) -> None:
        ai = self.engine()
        self.seed_support_failures(ai)
        useful = ai.run_learning_cycle()[0].patch_id
        dead = self._install_dead_rule(ai)
        version_before = ai.policy.version()

        findings = {finding.patch_id: finding for finding in ai.run_audit()}

        self.assertEqual(findings[useful].action, "kept")
        self.assertGreater(findings[useful].contribution, 0.3)
        self.assertEqual(findings[dead].action, "rolled_back")
        self.assertEqual(ai.db.get_patch(dead)["status"], "rolled_back")
        self.assertNotIn("Be polite.", ai.policy.render_system_prompt(route="support"))
        self.assertGreater(ai.policy.version(), version_before)
        self.assertEqual(ai.dashboard()["metrics"]["rolled_back_patches"], 1)

    def test_audit_without_prune_only_reports(self) -> None:
        ai = self.engine()
        self.seed_support_failures(ai)
        dead = self._install_dead_rule(ai)

        findings = {finding.patch_id: finding for finding in ai.run_audit(prune=False)}

        self.assertTrue(findings[dead].action.startswith("flagged"))
        self.assertEqual(ai.db.get_patch(dead)["status"], "promoted")

    def test_manual_rollback_restores_old_behavior(self) -> None:
        # Recall is off so the rule is the only thing shaping the answer.
        ai = self.engine(enable_recall=False)
        self.seed_support_failures(ai)
        patch_id = ai.run_learning_cycle()[0].patch_id
        prompt = "Customer says order #70555 never showed up and asks what happens now."
        self.assertIn("Resolution:", ai.generate(prompt).text)

        self.assertTrue(ai.rollback(patch_id, reason="testing"))

        self.assertNotIn("Resolution:", ai.generate(prompt).text)
        self.assertEqual(ai.db.get_patch(patch_id)["details"]["rollback_reason"], "testing")
        self.assertFalse(ai.rollback(patch_id))
