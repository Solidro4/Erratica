from __future__ import annotations

from typing import Optional

from erratica.storage.db import APPROVED, Database

UNSPECIFIED = "unspecified"


def normalize_issue_type(issue_type: Optional[str], rating: int) -> str:
    if issue_type and issue_type.strip():
        return issue_type.strip().lower().replace(" ", "_").replace("-", "_")
    return APPROVED if rating > 0 else UNSPECIFIED


class FeedbackIngestor:
    """Stores feedback exactly as given.

    Nothing is inferred from free-text notes. Required terms, the preferred
    format, and the reference answer are only what the caller passed in, so
    a comment like "too vague" can never turn into a demand that the answer
    contain the word "vague".
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def record(
        self,
        interaction_id: int,
        *,
        rating: int = -1,
        issue_type: Optional[str] = None,
        required_terms: Optional[list[str]] = None,
        preferred_format: Optional[str] = None,
        ideal_response: Optional[str] = None,
        correction: Optional[str] = None,
        notes: Optional[str] = None,
        source: str = "human",
    ) -> int:
        if self.db.get_interaction(interaction_id) is None:
            raise ValueError(f"no interaction with id {interaction_id}")
        return self.db.insert_feedback(
            interaction_id=interaction_id,
            rating=rating,
            issue_type=normalize_issue_type(issue_type, rating),
            required_terms=[term for term in (required_terms or []) if term and term.strip()],
            preferred_format=preferred_format,
            ideal_response=ideal_response,
            correction=correction,
            notes=notes,
            source=source,
        )
