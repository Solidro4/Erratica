from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from synora.learning.replay import ReplayCase
from synora.learning.similarity import HybridStringSimilarityScorer, SimilarityScorer


@dataclass
class EvaluationSummary:
    score_before: float
    score_after: float
    case_deltas: list[dict[str, Any]]

    @property
    def delta(self) -> float:
        return self.score_after - self.score_before


class PatchEvaluator:
    def __init__(self, *, similarity_scorer: SimilarityScorer | None = None) -> None:
        self.similarity_scorer = similarity_scorer or HybridStringSimilarityScorer()

    def evaluate(
        self,
        cases: list[ReplayCase],
        baseline_results: list[dict[str, Any]],
        candidate_results: list[dict[str, Any]],
    ) -> EvaluationSummary:
        before_scores: list[float] = []
        after_scores: list[float] = []
        case_deltas: list[dict[str, Any]] = []

        for case, before, after in zip(cases, baseline_results, candidate_results):
            before_score = self._score_case(case, before["response"])
            after_score = self._score_case(case, after["response"])
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
        )

    def _score_case(self, case: ReplayCase, response: str) -> float:
        checks: list[float] = []
        response_lower = response.lower()

        if case.required_terms:
            hits = sum(term.lower() in response_lower for term in case.required_terms)
            checks.append(hits / len(case.required_terms))

        if case.preferred_format:
            checks.append(1.0 if self._matches_format(case.preferred_format, response) else 0.0)

        if case.ideal_response:
            checks.append(self.similarity_scorer.score(case.ideal_response, response))

        issue_score = self._score_issue(case, response)
        if issue_score is not None:
            checks.append(issue_score)

        support_score = self._score_support_case(case, response)
        if support_score is not None:
            checks.append(support_score)

        return self._average(checks) if checks else 0.0

    def _score_issue(self, case: ReplayCase, response: str) -> float | None:
        issue_type = case.issue_type
        lowered = response.lower()
        if issue_type == "missing_segmentation":
            return 1.0 if self._looks_segmented(response) else 0.0
        if issue_type == "missing_next_steps":
            return 1.0 if "1." in response or "next step" in lowered else 0.0
        if issue_type == "missing_resolution":
            return 1.0 if self._has_resolution(lowered) and self._has_timeframe(lowered) else 0.0
        if issue_type == "missing_policy":
            return 1.0 if "policy" in lowered or self._has_timeframe(lowered) else 0.0
        if issue_type == "hallucinated_values":
            return 1.0 if self._numbers_grounded(case.prompt, response) else 0.0
        if issue_type == "missing_structure":
            return 1.0 if self._matches_format("bullet_list", response) or self._matches_format("numbered_list", response) else 0.0
        return None

    def _score_support_case(self, case: ReplayCase, response: str) -> float | None:
        if case.route != "support":
            return None

        response_lower = response.lower()
        checks: list[float] = []
        entities = self._expected_support_entities(case.prompt, case.required_terms)
        if entities:
            hits = sum(entity in response_lower for entity in entities)
            checks.append(hits / len(entities))
        checks.append(1.0 if self._has_resolution(response_lower) else 0.0)
        checks.append(1.0 if self._has_timeframe(response_lower) else 0.0)
        return self._average(checks)

    def _looks_segmented(self, response: str) -> bool:
        has_list = self._matches_format("bullet_list", response) or self._matches_format("numbered_list", response)
        has_multiple_entries = len(re.findall(r"(?m)^\s*(?:\d+\.\s+|- )?[A-Za-z][A-Za-z\s\-]+:", response)) >= 2
        return has_list and has_multiple_entries

    def _matches_format(self, preferred_format: str, response: str) -> bool:
        if preferred_format == "bullet_list":
            return bool(re.search(r"(?m)^\s*- ", response)) or "1." in response
        if preferred_format == "numbered_list":
            return bool(re.search(r"(?m)^\s*\d+\.\s+", response))
        if preferred_format == "sections":
            return response.count(":") >= 2
        if preferred_format == "paragraph":
            return "\n" not in response.strip()
        return False

    def _numbers_grounded(self, prompt: str, response: str) -> bool:
        prompt_numbers = set(re.findall(r"\$?[0-9]+(?:\.[0-9]+)?[MK]?", prompt))
        response_numbers = set(re.findall(r"\$?[0-9]+(?:\.[0-9]+)?[MK]?", response))
        return response_numbers.issubset(prompt_numbers)

    def _expected_support_entities(self, prompt: str, required_terms: list[str]) -> list[str]:
        expected: list[str] = []
        lowered = prompt.lower()
        if "refund" in lowered:
            expected.append("refund")
        if "order" in lowered:
            expected.append("order")
        if "charge" in lowered:
            expected.append("charge")
        if "billing" in lowered:
            expected.append("billing")
        if "shipment" in lowered or "tracking" in lowered:
            expected.append("shipment")
        if "replacement" in lowered or "damaged" in lowered:
            expected.append("replacement")
        for term in required_terms:
            normalized = term.lower()
            if normalized not in expected:
                expected.append(normalized)
        return expected[:4]

    def _has_resolution(self, response_lower: str) -> bool:
        return any(
            token in response_lower
            for token in (
                "resolution:",
                "refund",
                "reversing",
                "replacement",
                "escalating",
                "stopping renewal",
                "applying a concrete resolution",
            )
        )

    def _has_timeframe(self, response_lower: str) -> bool:
        return bool(re.search(r"\b(?:\d+\s*-\s*\d+|\d+)\s+(?:business days|hours|day)\b", response_lower))

    def _average(self, scores: list[float]) -> float:
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
