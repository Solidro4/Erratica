"""Small text helpers with no imports from the rest of the package."""

from __future__ import annotations

import re


def normalize_prompt(prompt: str) -> str:
    """Canonical form used to decide whether two prompts are the same."""
    return re.sub(r"\s+", " ", str(prompt).strip().lower())
