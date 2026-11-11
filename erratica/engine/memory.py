from __future__ import annotations

from typing import Any, Iterable, Optional

from erratica.learning.similarity import SimilarityScorer
from erratica.storage.db import Database
from erratica.textutil import normalize_prompt


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

    def failed_interactions(self, limit: int = 500) -> list[dict[str, Any]]:
        return self.db.list_failed_interactions(limit=limit)

    def similar_corrections(
        self,
        *,
        prompt: str,
        route: str,
        scorer: SimilarityScorer,
        limit: int = 2,
        min_score: float = 0.5,
        exclude_prompts: Iterable[str] = (),
        sources: Optional[Iterable[str]] = ("human",),
    ) -> list[dict[str, str]]:
        """Recall past approved answers for prompts similar to this one.

        This gives the model case-based experience at answer time: a
        correction is a candidate reference example the moment it is
        recorded, without waiting for a learning cycle.

        ``exclude_prompts`` removes specific past prompts from the pool.
        Replay uses it so a case is never scored with its own answer in
        front of the model.
        """
        excluded = {normalize_prompt(item) for item in exclude_prompts}
        corrections = self.db.list_corrections(route=route, sources=sources)
        scored: list[tuple[float, dict[str, str]]] = []
        seen_prompts: set[str] = set()
        for correction in corrections:
            past_prompt = correction["prompt"]
            key = normalize_prompt(past_prompt)
            if key in seen_prompts or key in excluded:
                continue
            seen_prompts.add(key)
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
