from __future__ import annotations

import re

from synora.storage.db import Database

_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "have",
    "into",
    "your",
    "about",
    "missed",
    "client",
    "clients",
    "please",
}


class FeedbackIngestor:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record(
        self,
        interaction_id: int,
        *,
        rating: int = -1,
        issue_type: str,
        required_terms: list[str] | None = None,
        preferred_format: str | None = None,
        ideal_response: str | None = None,
        correction: str | None = None,
        notes: str | None = None,
    ) -> int:
        normalized_issue = issue_type.strip().lower().replace(" ", "_")
        derived_terms = required_terms or self._extract_required_terms(correction or notes or "")
        return self.db.insert_feedback(
            interaction_id=interaction_id,
            rating=rating,
            issue_type=normalized_issue,
            required_terms=derived_terms,
            preferred_format=preferred_format,
            ideal_response=ideal_response,
            correction=correction,
            notes=notes,
        )

    def _extract_required_terms(self, text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
        seen: set[str] = set()
        extracted: list[str] = []
        for token in tokens:
            lowered = token.lower()
            if lowered in _STOPWORDS or lowered in seen:
                continue
            seen.add(lowered)
            extracted.append(token)
            if len(extracted) == 5:
                break
        return extracted
