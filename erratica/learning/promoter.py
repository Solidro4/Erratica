from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from erratica.domains import DomainRegistry
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
    regression_delta: Optional[float] = None
    reason: str = ""
    route: Optional[str] = None
    # "holdout": judged on cases the patch was not written from.
    # "in_sample": too few cases to hold any out; weak evidence.
    validation: str = "in_sample"
    cases_scored: int = 0


class PatchPromoter:
    """The gate every behavior change has to pass.

    A patch is promoted only if all of these hold:

    - it raises the mean score on its validation cases by more than
      ``min_delta``
    - it does not lower the mean score on answers a person approved by more
      than ``regression_tolerance``, and does not badly break any single
      one of them (``max_case_drop``)
    - it does not lower the mean score on the other failures of the same
      route by more than ``regression_tolerance``

    Domains can set stricter or looser gates for their own route.
    """

    def __init__(
        self,
        db: Database,
        policy: PromptPolicy,
        *,
        registry: Optional[DomainRegistry] = None,
        min_delta: float = 0.05,
        regression_tolerance: float = 0.02,
        max_case_drop: float = 0.3,
    ) -> None:
        self.db = db
        self.policy = policy
        self.registry = registry or DomainRegistry()
        self.min_delta = min_delta
        self.regression_tolerance = regression_tolerance
        self.max_case_drop = max_case_drop

    def gates_for(self, route: Optional[str]) -> tuple[float, float]:
        domain = self.registry.get(route or "general")
        min_delta = domain.min_delta if domain.min_delta is not None else self.min_delta
        tolerance = (
            domain.regression_tolerance
            if domain.regression_tolerance is not None
            else self.regression_tolerance
        )
        return min_delta, tolerance

    def promote(
        self,
        proposal: PatchProposal,
        evaluation: EvaluationSummary,
        *,
        regression: Optional[EvaluationSummary] = None,
        wins: Optional[EvaluationSummary] = None,
        validation: str = "in_sample",
        extra_details: Optional[dict[str, Any]] = None,
    ) -> PromotionDecision:
        """``regression`` is replay on other failures of the route; ``wins``
        is replay on answers a user approved. Both are optional because a
        fresh database has neither."""
        patch_id = self._record(proposal)
        min_delta, tolerance = self.gates_for(proposal.route)

        has_regression = regression is not None and regression.scored > 0
        has_wins = wins is not None and wins.scored > 0
        guard_deltas = [
            summary.delta
            for summary, present in ((regression, has_regression), (wins, has_wins))
            if present
        ]
        regression_delta = min(guard_deltas) if guard_deltas else None

        if evaluation.scored == 0:
            accepted = False
            reason = "no validation case had a checkable signal"
        elif evaluation.delta <= min_delta:
            accepted = False
            reason = "insufficient improvement on validation cases"
        elif has_wins and wins.delta < -tolerance:
            accepted = False
            reason = "regression on answers users approved"
        elif has_wins and wins.worst_case_delta < -self.max_case_drop:
            accepted = False
            reason = "broke an answer a user approved"
        elif has_regression and regression.delta < -tolerance:
            accepted = False
            reason = "regression on other failure cases"
        else:
            accepted = True
            reason = {
                "holdout": "improved held-out cases with no regression",
                "manual": "hand-written rule improved recorded failures with no regression",
            }.get(
                validation,
                "improved in-sample cases with no regression (too few cases to hold out)",
            )

        details: dict[str, Any] = {
            "reason": reason,
            "validation": validation,
            "cases_scored": evaluation.scored,
            "cases_unscorable": evaluation.unscorable,
            "cases_improved": evaluation.improved,
            "cases_worsened": evaluation.worsened,
            "regression_cases": regression.scored if has_regression else 0,
            "win_cases": wins.scored if has_wins else 0,
            "regression_delta": regression_delta,
            "min_delta": min_delta,
            "regression_tolerance": tolerance,
            "origin": proposal.origin,
        }
        if extra_details:
            details.update(extra_details)

        self.db.update_patch_status(
            patch_id=patch_id,
            status="promoted" if accepted else "rejected",
            score_before=evaluation.score_before,
            score_after=evaluation.score_after,
            details=details,
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
            route=proposal.route,
            validation=validation,
            cases_scored=evaluation.scored,
        )

    def reject(self, proposal: PatchProposal, reason: str) -> PromotionDecision:
        """Record a proposal that was thrown out before replay (for example
        a tool that failed static screening), so the audit trail is complete."""
        patch_id = self._record(proposal)
        self.db.update_patch_status(
            patch_id=patch_id,
            status="rejected",
            details={"reason": reason, "validation": "none", "origin": proposal.origin},
        )
        return PromotionDecision(
            patch_id=patch_id,
            accepted=False,
            score_before=0.0,
            score_after=0.0,
            delta=0.0,
            rule_text=proposal.rule_text,
            patch_type=proposal.patch_type,
            reason=reason,
            route=proposal.route,
            validation="none",
        )

    def rollback(self, patch_id: int, reason: str) -> bool:
        """Undo a promoted patch. Returns False if it was not live."""
        patch = self.db.get_patch(patch_id)
        if patch is None or patch["status"] != "promoted":
            return False
        self.db.deactivate_patch(patch_id)
        self.db.update_patch_status(
            patch_id=patch_id,
            status="rolled_back",
            details={"rollback_reason": reason},
        )
        return True

    def _record(self, proposal: PatchProposal) -> int:
        return self.db.insert_patch(
            patch_type=proposal.patch_type,
            target=proposal.target,
            content=proposal.content(),
            rationale=proposal.rationale,
            source_issue_type=proposal.source_issue_type,
            route=proposal.route,
        )

    def _apply(self, proposal: PatchProposal, patch_id: int) -> None:
        route = proposal.route or "general"
        if proposal.patch_type == "few_shot":
            self.policy.examples.add_example(
                route=route,
                prompt=proposal.example_prompt or "",
                response=proposal.example_response or "",
                source_patch_id=patch_id,
            )
        elif proposal.patch_type == "tool":
            self.policy.install_tool(
                name=proposal.tool_name or f"tool_{patch_id}",
                route=route,
                source=proposal.tool_source or "",
                source_patch_id=patch_id,
            )
        else:
            self.policy.apply_prompt_rule(
                proposal.rule_text,
                source_patch_id=patch_id,
                route=proposal.route,
            )
