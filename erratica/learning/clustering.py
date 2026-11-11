from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


def is_failing(case: dict[str, Any]) -> bool:
    """A case counts as a failure unless a user rated it positively."""
    return (case.get("rating") or 0) <= 0


@dataclass
class FailureCluster:
    issue_type: str
    cases: list[dict[str, Any]]
    required_terms: list[str]
    preferred_formats: list[str]
    route: str = "general"

    @property
    def size(self) -> int:
        return len(self.cases)

    @property
    def failing_cases(self) -> list[dict[str, Any]]:
        """Cases to replay. Positively rated rows are dissent, not failures."""
        return [case for case in self.cases if is_failing(case)]

    def subset(self, cases: list[dict[str, Any]]) -> "FailureCluster":
        """The same cluster restricted to some of its cases."""
        return build_cluster(self.issue_type, self.route, cases)


def build_cluster(issue_type: str, route: str, cases: list[dict[str, Any]]) -> FailureCluster:
    term_counter: Counter[str] = Counter()
    format_counter: Counter[str] = Counter()
    for case in cases:
        if not is_failing(case):
            continue
        for term in case.get("required_terms", []):
            term_counter[term] += 1
        preferred_format = case.get("preferred_format")
        if preferred_format:
            format_counter[preferred_format] += 1
    return FailureCluster(
        issue_type=issue_type,
        cases=cases,
        required_terms=[term for term, _ in term_counter.most_common(5)],
        preferred_formats=[name for name, _ in format_counter.most_common(3)],
        route=route,
    )


class FailureClusterer:
    """Groups failures that share a route and an issue type.

    Keeping routes apart means a fix is proposed, validated, and installed
    for one kind of task at a time.
    """

    def cluster(self, failures: list[dict[str, Any]]) -> list[FailureCluster]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for failure in failures:
            route = failure.get("route") or "general"
            issue_type = failure.get("issue_type") or "unspecified"
            grouped[(route, issue_type)].append(failure)

        clusters = [
            build_cluster(issue_type, route, cases)
            for (route, issue_type), cases in grouped.items()
        ]
        return sorted(clusters, key=lambda cluster: cluster.size, reverse=True)
