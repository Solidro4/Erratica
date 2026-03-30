from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from synora.learning.clustering import FailureCluster


@dataclass
class ReplayCase:
    interaction_id: int
    prompt: str
    baseline_response: str
    route: str
    issue_type: str
    required_terms: list[str]
    preferred_format: str | None
    ideal_response: str | None
    correction: str | None
    notes: str | None


class ReplayDatasetBuilder:
    def build(self, failures: list[dict[str, Any]]) -> list[ReplayCase]:
        return [self._to_case(failure) for failure in failures]

    def build_from_cluster(self, cluster: FailureCluster) -> list[ReplayCase]:
        return [self._to_case(case) for case in cluster.cases]

    def _to_case(self, failure: dict[str, Any]) -> ReplayCase:
        return ReplayCase(
            interaction_id=int(failure["interaction_id"]),
            prompt=str(failure["prompt"]),
            baseline_response=str(failure["response"]),
            route=str(failure["route"]),
            issue_type=str(failure["issue_type"]),
            required_terms=list(failure.get("required_terms", [])),
            preferred_format=failure.get("preferred_format"),
            ideal_response=failure.get("ideal_response"),
            correction=failure.get("correction"),
            notes=failure.get("notes"),
        )


class ReplayRunner:
    def __init__(self, model: Any, policy: Any, router: Any) -> None:
        self.model = model
        self.policy = policy
        self.router = router

    def run(
        self,
        cases: list[ReplayCase],
        *,
        extra_rules: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for case in cases:
            route = case.route or self.router.select_route(case.prompt)
            system_prompt = self.policy.render_system_prompt(route=route, extra_rules=extra_rules)
            response = self.model.generate(case.prompt, system_prompt, route)
            results.append(
                {
                    "interaction_id": case.interaction_id,
                    "route": route,
                    "response": response,
                }
            )
        return results
