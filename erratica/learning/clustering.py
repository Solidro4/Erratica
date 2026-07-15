from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class FailureCluster:
    issue_type: str
    cases: list[dict[str, Any]]
    required_terms: list[str]
    preferred_formats: list[str]

    @property
    def size(self) -> int:
        return len(self.cases)


class FailureClusterer:
    def cluster(self, failures: list[dict[str, Any]]) -> list[FailureCluster]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for failure in failures:
            key = failure.get("issue_type") or "unknown"
            grouped[key].append(failure)

        clusters: list[FailureCluster] = []
        for issue_type, cases in grouped.items():
            term_counter: Counter[str] = Counter()
            format_counter: Counter[str] = Counter()
            for case in cases:
                for term in case.get("required_terms", []):
                    term_counter[term] += 1
                preferred_format = case.get("preferred_format")
                if preferred_format:
                    format_counter[preferred_format] += 1
            clusters.append(
                FailureCluster(
                    issue_type=issue_type,
                    cases=cases,
                    required_terms=[term for term, _ in term_counter.most_common(5)],
                    preferred_formats=[name for name, _ in format_counter.most_common(3)],
                )
            )

        return sorted(clusters, key=lambda cluster: cluster.size, reverse=True)
