from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from erratica.domains import DomainRegistry, has_list
from erratica.learning.replay import ReplayCase
from erratica.learning.similarity import HybridStringSimilarityScorer, SimilarityScorer


@dataclass
class EvaluationSummary:
    score_before: float
    score_after: float
    case_deltas: list[dict[str, Any]]
    # Cases with no checkable signal are left out of the averages.
    unscorable: int = 0

    @property
    def delta(self) -> float:
        return self.score_after - self.score_before

    @property
    def scored(self) -> int:
        return len(self.case_deltas)

    @property
    def worst_case_delta(self) -> float:
        if not self.case_deltas:
            return 0.0
        return min(item["delta"] for item in self.case_deltas)

    @property
    def improved(self) -> int:
        return sum(1 for item in self.case_deltas if item["delta"] > 1e-9)

    @property
    def worsened(self) -> int:
        return sum(1 for item in self.case_deltas if item["delta"] < -1e-9)


class PatchEvaluator:
    """Scores an answer against everything known about what was wanted.

    Each available signal contributes one score in [0, 1]:

    - required terms the user listed
    - the format the user asked for
    - similarity to a reference answer, when one exists
    - the domain pack's checks for this route and issue type
    - any critics passed in (for example a larger local model as judge)

    The case score is the mean of the signals that apply. A case with no
    signal at all cannot be scored and is excluded rather than counted as
    zero, so missing evidence never looks like a failed patch or a win.
    """

    def __init__(
        self,
        *,
        similarity_scorer: Optional[SimilarityScorer] = None,
        registry: Optional[DomainRegistry] = None,
        critics: Optional[Sequence[Any]] = None,
    ) -> None:
        self.similarity_scorer = similarity_scorer or HybridStringSimilarityScorer()
        self.registry = registry or DomainRegistry()
        self.critics = list(critics or [])

    def evaluate(
        self,
        cases: list[ReplayCase],
        baseline_results: list[dict[str, Any]],
        candidate_results: list[dict[str, Any]],
    ) -> EvaluationSummary:
        before_scores: list[float] = []
        after_scores: list[float] = []
        case_deltas: list[dict[str, Any]] = []
        unscorable = 0

        for case, before, after in zip(cases, baseline_results, candidate_results):
            before_score = self.score_case(case, before["response"])
            after_score = self.score_case(case, after["response"])
            if before_score is None or after_score is None:
                unscorable += 1
                continue
            before_scores.append(before_score)
            after_scores.append(after_score)
            case_deltas.append(
                {
                    "interaction_id": case.interaction_id,
                    "before": before_score,
                    "after": after_score,
                    "delta": after_score - before_score,
                }
            )

        return EvaluationSummary(
            score_before=self._average(before_scores),
            score_after=self._average(after_scores),
            case_deltas=case_deltas,
            unscorable=unscorable,
        )

    def score_case(self, case: ReplayCase, response: str) -> Optional[float]:
        checks: list[float] = []
        response_lower = response.lower()

        if case.required_terms:
            hits = sum(term.lower() in response_lower for term in case.required_terms)
            checks.append(hits / len(case.required_terms))

        if case.preferred_format:
            matched = matches_format(case.preferred_format, response)
            if matched is not None:
                checks.append(1.0 if matched else 0.0)

        reference = case.reference
        if reference:
            checks.append(self.similarity_scorer.score(reference, response))

        for check in self.registry.checks_for(case.route, case.issue_type):
            value = check(case, response)
            if value is not None:
                checks.append(float(value))

        for critic in self.critics:
            value = critic.score(
                prompt=case.prompt,
                response=response,
                reference=reference,
                route=case.route,
            )
            if value is not None:
                checks.append(max(0.0, min(1.0, float(value))))

        if not checks:
            return None
        return self._average(checks)

    def _average(self, scores: list[float]) -> float:
        if not scores:
            return 0.0
        return sum(scores) / len(scores)


def matches_format(preferred_format: str, response: str) -> Optional[bool]:
    """True/False for a known format name, None for one we cannot check."""
    if preferred_format == "bullet_list":
        return has_list(response)
    if preferred_format == "numbered_list":
        return bool(re.search(r"(?m)^\s*\d+[.)]\s+", response))
    if preferred_format == "sections":
        return response.count(":") >= 2 or bool(re.search(r"(?m)^#+\s", response))
    if preferred_format == "paragraph":
        return "\n" not in response.strip() and not has_list(response)
    if preferred_format == "json":
        stripped = response.strip()
        return stripped.startswith(("{", "[")) and stripped.endswith(("}", "]"))
    if preferred_format == "code_block":
        return "```" in response
    return None
