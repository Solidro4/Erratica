from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.init_schema()

    def init_schema(self) -> None:
        self.connection.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.connection.commit()

    def insert_interaction(
        self,
        *,
        prompt: str,
        response: str,
        route: str,
        policy_version: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO interactions (prompt, response, route, policy_version, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (prompt, response, route, policy_version, self._dump_json(metadata)),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def list_recent_interactions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            """
            SELECT id, prompt, response, route, policy_version, created_at
            FROM interactions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def insert_feedback(
        self,
        *,
        interaction_id: int,
        rating: int,
        issue_type: str,
        required_terms: list[str] | None,
        preferred_format: str | None,
        correction: str | None,
        notes: str | None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO feedback (
                interaction_id, rating, issue_type, required_terms_json,
                preferred_format, correction, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interaction_id,
                rating,
                issue_type,
                self._dump_json(required_terms or []),
                preferred_format,
                correction,
                notes,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def list_failed_interactions(self, *, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            """
            SELECT
                i.id AS interaction_id,
                i.prompt,
                i.response,
                i.route,
                i.policy_version,
                f.id AS feedback_id,
                f.rating,
                f.issue_type,
                f.required_terms_json,
                f.preferred_format,
                f.correction,
                f.notes,
                f.created_at AS feedback_created_at
            FROM feedback AS f
            JOIN interactions AS i ON i.id = f.interaction_id
            WHERE COALESCE(f.rating, 0) < 0 OR f.issue_type IS NOT NULL
            ORDER BY f.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        failures: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            item = dict(row)
            item["required_terms"] = self._load_json(item.pop("required_terms_json"), [])
            failures.append(item)
        return failures

    def insert_patch(
        self,
        *,
        patch_type: str,
        target: str,
        content: dict[str, Any],
        rationale: str,
        source_issue_type: str,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO patches (
                patch_type, target, content_json, rationale, source_issue_type, status
            )
            VALUES (?, ?, ?, ?, ?, 'proposed')
            """,
            (patch_type, target, self._dump_json(content), rationale, source_issue_type),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def update_patch_status(
        self,
        *,
        patch_id: int,
        status: str,
        score_before: float,
        score_after: float,
    ) -> None:
        promoted_clause = "CURRENT_TIMESTAMP" if status == "promoted" else "NULL"
        self.connection.execute(
            f"""
            UPDATE patches
            SET status = ?, score_before = ?, score_after = ?, promoted_at = {promoted_clause}
            WHERE id = ?
            """,
            (status, score_before, score_after, patch_id),
        )
        self.connection.commit()

    def list_patch_history(self, *, limit: int = 10) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            """
            SELECT id, patch_type, target, content_json, rationale, source_issue_type,
                   status, score_before, score_after, created_at, promoted_at
            FROM patches
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        history: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            item = dict(row)
            item["content"] = self._load_json(item.pop("content_json"), {})
            history.append(item)
        return history

    def insert_policy_rule(
        self,
        *,
        rule_text: str,
        source_patch_id: int | None,
        active: bool,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO policy_rules (rule_text, source_patch_id, active, activated_at)
            VALUES (?, ?, ?, CASE WHEN ? = 1 THEN CURRENT_TIMESTAMP ELSE NULL END)
            """,
            (rule_text, source_patch_id, int(active), int(active)),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def list_policy_rules(self, *, active_only: bool = True) -> list[dict[str, Any]]:
        if active_only:
            cursor = self.connection.execute(
                """
                SELECT id, rule_text, source_patch_id, active, created_at, activated_at
                FROM policy_rules
                WHERE active = 1
                ORDER BY id ASC
                """
            )
        else:
            cursor = self.connection.execute(
                """
                SELECT id, rule_text, source_patch_id, active, created_at, activated_at
                FROM policy_rules
                ORDER BY id ASC
                """
            )
        return [dict(row) for row in cursor.fetchall()]

    def get_policy_version(self) -> int:
        cursor = self.connection.execute(
            "SELECT COUNT(*) AS count FROM policy_rules WHERE active = 1"
        )
        row = cursor.fetchone()
        return int(row["count"]) if row else 0

    def insert_few_shot_example(
        self,
        *,
        route: str,
        prompt: str,
        response: str,
        active: bool,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO few_shot_examples (route, prompt, response, active)
            VALUES (?, ?, ?, ?)
            """,
            (route, prompt, response, int(active)),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def list_few_shot_examples(self, *, route: str, limit: int = 2) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            """
            SELECT id, route, prompt, response
            FROM few_shot_examples
            WHERE active = 1 AND route = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (route, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_dashboard_snapshot(self) -> dict[str, Any]:
        metrics = {
            "interactions": self._count("interactions"),
            "feedback_items": self._count("feedback"),
            "promoted_patches": self._count("patches", "status = 'promoted'"),
            "active_rules": self._count("policy_rules", "active = 1"),
        }
        issue_counts = self.connection.execute(
            """
            SELECT issue_type, COUNT(*) AS count
            FROM feedback
            GROUP BY issue_type
            ORDER BY count DESC, issue_type ASC
            """
        ).fetchall()
        return {
            "metrics": metrics,
            "issue_clusters": [dict(row) for row in issue_counts],
            "active_rules": self.list_policy_rules(active_only=True),
            "recent_patches": self.list_patch_history(limit=10),
            "recent_interactions": self.list_recent_interactions(limit=5),
        }

    def close(self) -> None:
        self.connection.close()

    def _count(self, table: str, where: str | None = None) -> int:
        query = f"SELECT COUNT(*) AS count FROM {table}"
        if where:
            query += f" WHERE {where}"
        row = self.connection.execute(query).fetchone()
        return int(row["count"]) if row else 0

    def _dump_json(self, value: Any) -> str:
        if value is None:
            return json.dumps({})
        return json.dumps(value)

    def _load_json(self, value: str | None, default: Any) -> Any:
        if not value:
            return default
        return json.loads(value)
