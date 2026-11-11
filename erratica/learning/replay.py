from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Optional

from erratica.learning.clustering import FailureCluster

# (prompt, route) -> reference examples recalled for that prompt, never
# including the prompt's own correction.
RecallFn = Callable[[str, str], list]


@dataclass
class ReplayCase:
    interaction_id: int
    prompt: str
    baseline_response: str
    route: str
    issue_type: str
    required_terms: list[str] = field(default_factory=list)
    preferred_format: Optional[str] = None
    ideal_response: Optional[str] = None
    correction: Optional[str] = None
    notes: Optional[str] = None
    source: str = "human"

    @property
    def reference(self) -> Optional[str]:
        return self.ideal_response or self.correction or None


class ReplayDatasetBuilder:
    def build(self, failures: list[dict[str, Any]]) -> list[ReplayCase]:
        return [self._to_case(failure) for failure in failures]

    def build_from_cluster(self, cluster: FailureCluster) -> list[ReplayCase]:
        return [self._to_case(case) for case in cluster.failing_cases]

    def _to_case(self, failure: dict[str, Any]) -> ReplayCase:
        return ReplayCase(
            interaction_id=int(failure["interaction_id"]),
            prompt=str(failure["prompt"]),
            baseline_response=str(failure.get("response", "")),
            route=str(failure.get("route") or "general"),
            issue_type=str(failure.get("issue_type") or "unspecified"),
            required_terms=list(failure.get("required_terms") or []),
            preferred_format=failure.get("preferred_format"),
            ideal_response=failure.get("ideal_response"),
            correction=failure.get("correction"),
            notes=failure.get("notes"),
            source=str(failure.get("source") or "human"),
        )


class ReplayRunner:
    """Re-asks past prompts under a candidate policy.

    Replay builds the system prompt the same way live generation does,
    including experience recall, so a patch is measured on what it adds to
    the system as deployed rather than to a stripped-down version of it.
    """

    def __init__(
        self,
        model: Any,
        policy: Any,
        router: Any,
        *,
        recall: Optional[RecallFn] = None,
    ) -> None:
        self.model = model
        self.policy = policy
        self.router = router
        self.recall = recall

    def run(
        self,
        cases: list[ReplayCase],
        *,
        extra_rules: Optional[list[str]] = None,
        extra_examples: Optional[list[dict[str, str]]] = None,
        extra_tools: Optional[list[dict[str, str]]] = None,
        exclude_patch_ids: Collection[int] = (),
        model: Any = None,
    ) -> list[dict[str, Any]]:
        active_model = model or self.model
        results: list[dict[str, Any]] = []
        for case in cases:
            route = case.route or self.router.select_route(case.prompt)
            examples = list(self.recall(case.prompt, route)) if self.recall else []
            if extra_examples:
                examples = examples + [
                    example for example in extra_examples if example["prompt"] != case.prompt
                ]
            system_prompt = self.policy.render_system_prompt(
                route=route,
                prompt=case.prompt,
                extra_rules=extra_rules,
                extra_examples=examples or None,
                extra_tools=extra_tools,
                exclude_patch_ids=exclude_patch_ids,
            )
            response = active_model.generate(case.prompt, system_prompt, route)
            results.append(
                {
                    "interaction_id": case.interaction_id,
                    "route": route,
                    "response": response,
                }
            )
        return results
