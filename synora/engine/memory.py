from __future__ import annotations

from typing import Any

from synora.storage.db import Database


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
