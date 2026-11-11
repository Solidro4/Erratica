from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from erratica.storage.schema import MIGRATIONS, SCHEMA

# Feedback rows that only say "this answer was good" carry this issue type.
APPROVED = "approved"


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.init_schema()

    def init_schema(self) -> None:
        self.connection.executescript(SCHEMA)
        for table, column, column_type in MIGRATIONS:
            self._ensure_column(table, column, column_type)
        self.connection.commit()

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------

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

    def get_interaction(self, interaction_id: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT id, prompt, response, route, policy_version, created_at FROM interactions WHERE id = ?",
            (interaction_id,),
        ).fetchone()
        return dict(row) if row else None

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

    def list_unreviewed_interactions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Interactions nobody (human or critic) has given feedback on yet."""
        cursor = self.connection.execute(
            """
            SELECT i.id, i.prompt, i.response, i.route
            FROM interactions AS i
            WHERE NOT EXISTS (SELECT 1 FROM feedback AS f WHERE f.interaction_id = i.id)
            ORDER BY i.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def insert_feedback(
        self,
        *,
        interaction_id: int,
        rating: int,
        issue_type: str,
        required_terms: list[str] | None,
        preferred_format: str | None,
        ideal_response: str | None,
        correction: str | None,
        notes: str | None,
        source: str = "human",
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO feedback (
                interaction_id, rating, issue_type, required_terms_json,
                preferred_format, ideal_response, correction, notes, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interaction_id,
                rating,
                issue_type,
                self._dump_json(required_terms or []),
                preferred_format,
                ideal_response,
                correction,
                notes,
                source,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def list_failed_interactions(self, *, limit: int = 500) -> list[dict[str, Any]]:
        """Every feedback row that reports a problem.

        Rows with a positive rating are included when they name an issue
        type: they are dissenting votes, and triage needs to see them. Plain
        approvals (issue type 'approved') are wins, not failures.
        """
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
                f.ideal_response,
                f.correction,
                f.notes,
                f.source,
                f.created_at AS feedback_created_at
            FROM feedback AS f
            JOIN interactions AS i ON i.id = f.interaction_id
            WHERE f.issue_type != ?
            ORDER BY f.id DESC
            LIMIT ?
            """,
            (APPROVED, limit),
        )
        return [self._feedback_row(row) for row in cursor.fetchall()]

    def list_wins(
        self,
        *,
        route: str | None = None,
        limit: int = 200,
        sources: Iterable[str] | None = ("human",),
    ) -> list[dict[str, Any]]:
        """Interactions a person approved. These are what a patch must not break."""
        query = """
            SELECT
                i.id AS interaction_id,
                i.prompt,
                i.response,
                i.route,
                f.rating,
                f.required_terms_json,
                f.preferred_format,
                f.source,
                COALESCE(NULLIF(f.ideal_response, ''), NULLIF(f.correction, ''), i.response)
                    AS ideal_response
            FROM feedback AS f
            JOIN interactions AS i ON i.id = f.interaction_id
            WHERE COALESCE(f.rating, 0) > 0
        """
        params: list[Any] = []
        if route is not None:
            query += " AND i.route = ?"
            params.append(route)
        if sources is not None:
            source_list = list(sources)
            query += " AND f.source IN (%s)" % ",".join("?" for _ in source_list)
            params.extend(source_list)
        query += " ORDER BY f.id DESC LIMIT ?"
        params.append(limit)
        rows = []
        for row in self.connection.execute(query, params).fetchall():
            item = self._feedback_row(row)
            item["issue_type"] = APPROVED
            rows.append(item)
        return rows

    def list_corrections(
        self,
        *,
        route: str | None = None,
        limit: int = 500,
        sources: Iterable[str] | None = ("human",),
    ) -> list[dict[str, Any]]:
        """Reference answers attached to feedback (ideal responses and corrections).

        Only human-written references are returned unless other sources are
        asked for: a correction a critic model wrote is a hypothesis, not
        ground truth.
        """
        query = """
            SELECT
                i.id AS interaction_id,
                i.prompt,
                i.route,
                f.source,
                COALESCE(NULLIF(f.ideal_response, ''), NULLIF(f.correction, '')) AS reference_response
            FROM feedback AS f
            JOIN interactions AS i ON i.id = f.interaction_id
            WHERE COALESCE(NULLIF(f.ideal_response, ''), NULLIF(f.correction, '')) IS NOT NULL
        """
        params: list[Any] = []
        if route is not None:
            query += " AND i.route = ?"
            params.append(route)
        if sources is not None:
            source_list = list(sources)
            query += " AND f.source IN (%s)" % ",".join("?" for _ in source_list)
            params.extend(source_list)
        query += " ORDER BY f.id DESC LIMIT ?"
        params.append(limit)
        cursor = self.connection.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Patches (the audit trail)
    # ------------------------------------------------------------------

    def insert_patch(
        self,
        *,
        patch_type: str,
        target: str,
        content: dict[str, Any],
        rationale: str,
        source_issue_type: str,
        route: str | None = None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO patches (
                patch_type, target, content_json, rationale, source_issue_type, route, status
            )
            VALUES (?, ?, ?, ?, ?, ?, 'proposed')
            """,
            (patch_type, target, self._dump_json(content), rationale, source_issue_type, route),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def update_patch_status(
        self,
        *,
        patch_id: int,
        status: str,
        score_before: float | None = None,
        score_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        assignments = ["status = ?"]
        params: list[Any] = [status]
        if score_before is not None:
            assignments.append("score_before = ?")
            params.append(score_before)
        if score_after is not None:
            assignments.append("score_after = ?")
            params.append(score_after)
        if details is not None:
            existing = self.get_patch(patch_id)
            merged = dict(existing["details"]) if existing else {}
            merged.update(details)
            assignments.append("details_json = ?")
            params.append(self._dump_json(merged))
        if status == "promoted":
            assignments.append("promoted_at = COALESCE(promoted_at, CURRENT_TIMESTAMP)")
        params.append(patch_id)
        self.connection.execute(
            f"UPDATE patches SET {', '.join(assignments)} WHERE id = ?",
            params,
        )
        self.connection.commit()

    def get_patch(self, patch_id: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            f"SELECT {self._PATCH_COLUMNS} FROM patches WHERE id = ?",
            (patch_id,),
        ).fetchone()
        return self._patch_row(row) if row else None

    def list_patch_history(
        self,
        *,
        limit: int = 10,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        query = f"SELECT {self._PATCH_COLUMNS} FROM patches"
        params: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [self._patch_row(row) for row in self.connection.execute(query, params).fetchall()]

    def deactivate_patch(self, patch_id: int) -> int:
        """Turn off everything a patch installed. Returns how many rows changed."""
        changed = 0
        for table in ("policy_rules", "few_shot_examples", "tools"):
            cursor = self.connection.execute(
                f"UPDATE {table} SET active = 0 WHERE source_patch_id = ? AND active = 1",
                (patch_id,),
            )
            changed += cursor.rowcount
        self.connection.commit()
        return changed

    def get_policy_version(self) -> int:
        """Counts every change to live behavior: each promotion and each rollback."""
        row = self.connection.execute(
            """
            SELECT
                SUM(CASE WHEN status IN ('promoted', 'rolled_back') THEN 1 ELSE 0 END)
                + SUM(CASE WHEN status = 'rolled_back' THEN 1 ELSE 0 END) AS version
            FROM patches
            """
        ).fetchone()
        return int(row["version"] or 0) if row else 0

    # ------------------------------------------------------------------
    # Live policy: rules, examples, tools
    # ------------------------------------------------------------------

    def insert_policy_rule(
        self,
        *,
        rule_text: str,
        source_patch_id: int | None,
        active: bool,
        route: str | None = None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO policy_rules (rule_text, source_patch_id, active, route, activated_at)
            VALUES (?, ?, ?, ?, CASE WHEN ? = 1 THEN CURRENT_TIMESTAMP ELSE NULL END)
            """,
            (rule_text, source_patch_id, int(active), route, int(active)),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def list_policy_rules(
        self,
        *,
        active_only: bool = True,
        route: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rules for a route. A rule with no route applies everywhere."""
        query = """
            SELECT id, rule_text, source_patch_id, active, route, created_at, activated_at
            FROM policy_rules
            WHERE 1 = 1
        """
        params: list[Any] = []
        if active_only:
            query += " AND active = 1"
        if route is not None:
            query += " AND (route IS NULL OR route = ?)"
            params.append(route)
        query += " ORDER BY id ASC"
        return [dict(row) for row in self.connection.execute(query, params).fetchall()]

    def insert_few_shot_example(
        self,
        *,
        route: str,
        prompt: str,
        response: str,
        active: bool,
        source_patch_id: int | None = None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO few_shot_examples (route, prompt, response, active, source_patch_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (route, prompt, response, int(active), source_patch_id),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def list_few_shot_examples(self, *, route: str, limit: int = 2) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            """
            SELECT id, route, prompt, response, source_patch_id
            FROM few_shot_examples
            WHERE active = 1 AND route = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (route, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def list_all_few_shot_examples(self) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            """
            SELECT id, route, prompt, response, source_patch_id
            FROM few_shot_examples
            WHERE active = 1
            ORDER BY id ASC
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def insert_tool(
        self,
        *,
        name: str,
        route: str,
        source: str,
        source_patch_id: int | None = None,
        active: bool = True,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO tools (name, route, source, active, source_patch_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, route, source, int(active), source_patch_id),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def list_tools(self, *, route: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT id, name, route, source, source_patch_id FROM tools WHERE active = 1"
        params: list[Any] = []
        if route is not None:
            query += " AND route = ?"
            params.append(route)
        query += " ORDER BY id ASC"
        return [dict(row) for row in self.connection.execute(query, params).fetchall()]

    # ------------------------------------------------------------------
    # Triage
    # ------------------------------------------------------------------

    def insert_triage(
        self,
        *,
        issue_type: str,
        verdict: str,
        score: float,
        reasons: list[str],
        case_count: int,
        route: str | None = None,
        max_feedback_id: int | None = None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO mistake_triage (
                issue_type, verdict, score, reasons_json, case_count, route, max_feedback_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                issue_type,
                verdict,
                score,
                self._dump_json(reasons),
                case_count,
                route,
                max_feedback_id,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def last_reviewed_feedback_id(self, *, route: str, issue_type: str) -> int:
        """Newest feedback id a learning cycle has already looked at for this cluster."""
        row = self.connection.execute(
            """
            SELECT MAX(max_feedback_id) AS last_id
            FROM mistake_triage
            WHERE issue_type = ? AND COALESCE(route, '') = ?
            """,
            (issue_type, route),
        ).fetchone()
        return int(row["last_id"] or 0) if row else 0

    def list_triage(self, *, limit: int = 10) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            """
            SELECT id, issue_type, verdict, score, reasons_json, case_count, route, created_at
            FROM mistake_triage
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        entries: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            item = dict(row)
            item["reasons"] = self._load_json(item.pop("reasons_json"), [])
            entries.append(item)
        return entries

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def get_dashboard_snapshot(self) -> dict[str, Any]:
        metrics = {
            "interactions": self._count("interactions"),
            "feedback_items": self._count("feedback"),
            "approvals": self._count("feedback", "COALESCE(rating, 0) > 0 AND source = 'human'"),
            "self_review_items": self._count("feedback", "source != 'human'"),
            "promoted_patches": self._count("patches", "status = 'promoted'"),
            "rejected_patches": self._count("patches", "status = 'rejected'"),
            "rolled_back_patches": self._count("patches", "status = 'rolled_back'"),
            "active_rules": self._count("policy_rules", "active = 1"),
            "active_examples": self._count("few_shot_examples", "active = 1"),
            "active_tools": self._count("tools", "active = 1"),
            "quarantined_clusters": self._count("mistake_triage", "verdict = 'quarantined'"),
            "policy_version": self.get_policy_version(),
        }
        issue_counts = self.connection.execute(
            """
            SELECT issue_type, COUNT(*) AS count
            FROM feedback
            WHERE issue_type != ?
            GROUP BY issue_type
            ORDER BY count DESC, issue_type ASC
            """,
            (APPROVED,),
        ).fetchall()
        return {
            "metrics": metrics,
            "issue_clusters": [dict(row) for row in issue_counts],
            "active_rules": self.list_policy_rules(active_only=True),
            "active_tools": self.list_tools(),
            "recent_patches": self.list_patch_history(limit=10),
            "recent_interactions": self.list_recent_interactions(limit=5),
            "recent_triage": self.list_triage(limit=10),
        }

    def close(self) -> None:
        self.connection.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _PATCH_COLUMNS = (
        "id, patch_type, target, content_json, rationale, source_issue_type, status, "
        "score_before, score_after, route, details_json, created_at, promoted_at"
    )

    def _patch_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["content"] = self._load_json(item.pop("content_json"), {})
        item["details"] = self._load_json(item.pop("details_json"), {})
        return item

    def _feedback_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["required_terms"] = self._load_json(item.pop("required_terms_json", None), [])
        return item

    def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        rows = self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row["name"] for row in rows}
        if column in existing:
            return
        self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

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
