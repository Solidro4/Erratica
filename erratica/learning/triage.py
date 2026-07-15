from __future__ import annotations

import math
from dataclasses import dataclass

from erratica.learning.clustering import FailureCluster

_LIST_FORMATS = {"bullet_list", "numbered_list"}

# Weights of each evidence channel in the learnability score.
_WEIGHT_REFERENCE = 0.35
_WEIGHT_FAILURE_CONFIDENCE = 0.30
_WEIGHT_CONSISTENCY = 0.15
_WEIGHT_RECURRENCE = 0.20

_LEARNABLE_THRESHOLD = 0.45

# z = 1.0 gives an ~84% one-sided confidence lower bound.
_WILSON_Z = 1.0


@dataclass
class TriageVerdict:
    learnable: bool
    score: float
    reasons: list[str]

    @property
    def verdict(self) -> str:
        return "learnable" if self.learnable else "quarantined"


class MistakeTriage:
    """Separates mistakes worth learning from noise, using pure math.

    Each failure cluster is reduced to four evidence channels in a single
    O(n) pass with no model calls and no external dependencies:

    - reference strength: fraction of cases carrying a user-approved answer
    - failure confidence: Wilson score lower bound on the proportion of
      negative ratings, so small samples cannot fake certainty
    - consistency: 1 minus the normalized Shannon entropy of the requested
      formats, so contradictory feedback scores low
    - recurrence: saturating n / (n + 2) mass, so repeated failures matter
      more but a flood of duplicates cannot dominate

    The channels combine into a weighted learnability score. Contradictory
    evidence (conflicting formats, or ratings that disagree the behavior is
    even a failure) is a hard quarantine regardless of score. Quarantined
    clusters stay on record and become learnable once more evidence arrives.
    """

    def review(self, cluster: FailureCluster) -> TriageVerdict:
        reasons: list[str] = []
        size = cluster.size
        if size == 0:
            return TriageVerdict(learnable=False, score=0.0, reasons=["empty cluster"])

        reference_count = 0
        negative = 0
        positive = 0
        format_counts: dict[str, int] = {}
        for case in cluster.cases:
            if case.get("ideal_response") or case.get("correction"):
                reference_count += 1
            rating = case.get("rating") or 0
            if rating < 0:
                negative += 1
            elif rating > 0:
                positive += 1
            preferred_format = case.get("preferred_format")
            if preferred_format:
                format_counts[preferred_format] = format_counts.get(preferred_format, 0) + 1

        # Hard contradiction gates: no score can rescue self-contradicting evidence.
        if "paragraph" in format_counts and _LIST_FORMATS & format_counts.keys():
            reasons.append("feedback demands conflicting formats (paragraph vs list)")
            return TriageVerdict(learnable=False, score=0.0, reasons=reasons)
        if positive and positive >= negative:
            reasons.append("ratings disagree that the behavior is a failure")
            return TriageVerdict(learnable=False, score=0.0, reasons=reasons)

        reference_strength = reference_count / size
        failure_confidence = self._wilson_lower_bound(negative, negative + positive)
        consistency = 1.0 - self._normalized_entropy(list(format_counts.values()))
        recurrence = size / (size + 2.0)

        score = (
            _WEIGHT_REFERENCE * reference_strength
            + _WEIGHT_FAILURE_CONFIDENCE * failure_confidence
            + _WEIGHT_CONSISTENCY * consistency
            + _WEIGHT_RECURRENCE * recurrence
        )

        reasons.append(f"reference strength {reference_strength:.2f}")
        reasons.append(f"failure confidence {failure_confidence:.2f} (Wilson lower bound)")
        reasons.append(f"feedback consistency {consistency:.2f} (entropy-based)")
        reasons.append(f"recurrence mass {recurrence:.2f} over {size} case(s)")

        learnable = score >= _LEARNABLE_THRESHOLD
        reasons.append(
            f"learnability score {score:.2f} {'>=' if learnable else '<'} threshold {_LEARNABLE_THRESHOLD:.2f}"
        )
        return TriageVerdict(learnable=learnable, score=score, reasons=reasons)

    def _wilson_lower_bound(self, successes: int, trials: int) -> float:
        if trials == 0:
            return 0.0
        z = _WILSON_Z
        z_squared = z * z
        proportion = successes / trials
        denominator = 1.0 + z_squared / trials
        center = proportion + z_squared / (2.0 * trials)
        margin = z * math.sqrt(
            (proportion * (1.0 - proportion) + z_squared / (4.0 * trials)) / trials
        )
        return max(0.0, (center - margin) / denominator)

    def _normalized_entropy(self, counts: list[int]) -> float:
        if len(counts) <= 1:
            return 0.0
        total = float(sum(counts))
        entropy = -sum(
            (count / total) * math.log(count / total)
            for count in counts
            if count > 0
        )
        return entropy / math.log(len(counts))
