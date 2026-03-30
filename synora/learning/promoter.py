from __future__ import annotations

from dataclasses import dataclass

from synora.learning.evaluator import EvaluationSummary
from synora.learning.patcher import PatchProposal
from synora.policy.prompt_rules import PromptPolicy
from synora.storage.db import Database


@dataclass
class PromotionDecision:
    patch_id: int
    accepted: bool
    score_before: float
    score_after: float
    delta: float
    rule_text: str


class PatchPromoter:
    def __init__(
        self,
        db: Database,
        policy: PromptPolicy,
        *,
        min_delta: float = 0.05,
    ) -> None:
        self.db = db
        self.policy = policy
        self.min_delta = min_delta

    def promote(
        self,
        proposal: PatchProposal,
        evaluation: EvaluationSummary,
    ) -> PromotionDecision:
        patch_id = self.db.insert_patch(
            patch_type=proposal.patch_type,
            target=proposal.target,
            content={"rule_text": proposal.rule_text},
            rationale=proposal.rationale,
            source_issue_type=proposal.source_issue_type,
        )
        accepted = evaluation.delta > self.min_delta
        status = "promoted" if accepted else "rejected"
        self.db.update_patch_status(
            patch_id=patch_id,
            status=status,
            score_before=evaluation.score_before,
            score_after=evaluation.score_after,
        )
        if accepted:
            self.policy.apply_prompt_rule(proposal.rule_text, source_patch_id=patch_id)

        return PromotionDecision(
            patch_id=patch_id,
            accepted=accepted,
            score_before=evaluation.score_before,
            score_after=evaluation.score_after,
            delta=evaluation.delta,
            rule_text=proposal.rule_text,
        )
