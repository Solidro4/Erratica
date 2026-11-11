from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional

from erratica import Erratica
from erratica.cli.common import load_cases, record_cases

REFUND_PROMPT = "Customer says order #41327 was returned two weeks ago and the refund still has not arrived."
REFUND_IDEAL = (
    "I reviewed your request about your refund request.\n"
    "1. Reference: order 41327\n"
    "2. Resolution: we are approving the refund back to the original payment method\n"
    "3. Timeline: 3-5 business days\n"
    "4. Next step: we will send a confirmation update as soon as the action is complete."
)
VAGUE_ANSWER = "Thanks for reaching out. We are reviewing the issue and will follow up soon."


class EngineTestCase(unittest.TestCase):
    """Gives each test a throwaway database directory."""

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.tmp = Path(self._tempdir.name)

    def engine(self, **kwargs: Any) -> Erratica:
        ai = Erratica(db_path=self.tmp / kwargs.pop("db_name", "erratica.db"), **kwargs)
        self.addCleanup(ai.close)
        return ai

    def seed_support_failures(self, ai: Erratica, count: Optional[int] = None) -> list[dict]:
        """Record the bundled support failures. Returns the ones that failed."""
        cases = load_cases()[:count]
        recorded = record_cases(ai, cases)
        return [case for case, item in zip(cases, recorded) if item.failed]


def report_failure(ai: Erratica, prompt: str, *, route: Optional[str] = None, **feedback: Any) -> int:
    """Generate an answer and record negative feedback on it."""
    result = ai.generate(prompt, route=route)
    feedback.setdefault("issue_type", "test_issue")
    ai.learn_from_feedback(interaction_id=result.interaction_id, rating=-1, **feedback)
    return result.interaction_id
