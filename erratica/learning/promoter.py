from __future__ import annotations

from dataclasses import dataclass

from erratica.learning.evaluator import EvaluationSummary
from erratica.learning.patcher import PatchProposal
from erratica.policy.prompt_rules import PromptPolicy
from erratica.storage.db import Database


@dataclass
class PromotionDecision:
    patch_id: int
    accepted: bool
    score_before: float
    score_after: float
    delta: float
    rule_text: str
    patch_type: str = "prompt_rule"
    regression_delta: float | None = None
    reason: str = ""


class PatchPromoter:
    def __init__(
        self,
        db: Database,
        policy: PromptPolicy,
        *,
        min_delta: float = 0.05,
        regression_tolerance: float = 0.02,
    ) -> None:
        self.db = db
        self.policy = policy
        self.min_delta = min_delta
        self.regression_tolerance = regression_tolerance

    def promote(
        self,
        proposal: PatchProposal,
        evaluation: EvaluationSummary,
        *,
        regression: EvaluationSummary | None = None,
    ) -> PromotionDecision:
        patch_id = self.db.insert_patch(
            patch_type=proposal.patch_type,
            target=proposal.target,
            content=proposal.content(),
            rationale=proposal.rationale,
            source_issue_type=proposal.source_issue_type,
        )

        improved = evaluation.delta > self.min_delta
        regression_delta = regression.delta if regression is not None else None
        regressed = (
            regression_delta is not None
            and regression_delta < -self.regression_tolerance
        )
        accepted = improved and not regressed

        if not improved:
            reason = "insufficient improvement on replayed failures"
        elif regressed:
            reason = "regression on previously learned cases"
        else:
            reason = "validated improvement with no regression"

        status = "promoted" if accepted else "rejected"
        self.db.update_patch_status(
            patch_id=patch_id,
            status=status,
            score_before=evaluation.score_before,
            score_after=evaluation.score_after,
        )
        if accepted:
            self._apply(proposal, patch_id)

        return PromotionDecision(
            patch_id=patch_id,
            accepted=accepted,
            score_before=evaluation.score_before,
            score_after=evaluation.score_after,
            delta=evaluation.delta,
            rule_text=proposal.rule_text,
            patch_type=proposal.patch_type,
            regression_delta=regression_delta,
            reason=reason,
        )

    def _apply(self, proposal: PatchProposal, patch_id: int) -> None:
        if proposal.patch_type == "few_shot":
            self.policy.examples.add_example(
                route=proposal.route or "general",
                prompt=proposal.example_prompt or "",
                response=proposal.example_response or "",
            )
            return
        self.policy.apply_prompt_rule(proposal.rule_text, source_patch_id=patch_id)
