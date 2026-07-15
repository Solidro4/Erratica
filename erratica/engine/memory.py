from __future__ import annotations

from typing import Any

from erratica.learning.similarity import SimilarityScorer
from erratica.storage.db import Database


class MemoryStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record_interaction(
        self,
        prompt: str,
        response: str,
        route: str,
        policy_version: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        return self.db.insert_interaction(
            prompt=prompt,
            response=response,
            route=route,
            policy_version=policy_version,
            metadata=metadata,
        )

    def recent_interactions(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.db.list_recent_interactions(limit=limit)

    def failed_interactions(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.db.list_failed_interactions(limit=limit)

    def similar_corrections(
        self,
        *,
        prompt: str,
        route: str,
        scorer: SimilarityScorer,
        limit: int = 2,
        min_score: float = 0.35,
    ) -> list[dict[str, str]]:
        """Recall past user-approved answers for prompts similar to this one.

        This gives the model case-based experience at answer time: every
        correction it has ever received is a candidate reference example,
        without waiting for a learning cycle or retraining.
        """
        corrections = self.db.list_corrections(route=route)
        scored: list[tuple[float, dict[str, str]]] = []
        seen_prompts: set[str] = set()
        for correction in corrections:
            past_prompt = correction["prompt"]
            if past_prompt in seen_prompts:
                continue
            seen_prompts.add(past_prompt)
            score = scorer.score(past_prompt, prompt)
            if score < min_score:
                continue
            scored.append(
                (
                    score,
                    {
                        "prompt": past_prompt,
                        "response": correction["reference_response"],
                    },
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        return [example for _, example in scored[:limit]]
