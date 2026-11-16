"""Checks: how an answer is scored.

A check is any function ``(case, output) -> score``. The score is a number
from 0 to 1, or ``True`` / ``False``, or ``None`` when the check has nothing
to say about this case. A case's score is the mean of the checks that apply;
a case no check applies to is left out of every average instead of being
counted as zero.

Checks are code, cases are data: a check reads what it needs from the case
(``case.expect``, ``case.must``, ``case.meta``), so the same check works on
every case recorded before or after it was written.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable, List, Optional, Sequence

from erratica.core.types import APPROVED, Case, Check
from erratica.learning.similarity import HybridStringSimilarityScorer


def score_case(case: Case, output: Any, checks: Sequence[Check]) -> Optional[float]:
    """Mean of the applicable checks, or None when none applies."""
    values: List[float] = []
    for check in checks:
        issues = getattr(check, "issues", None)
        if issues and case.issue not in issues:
            continue
        value = check(case, output)
        if value is None:
            continue
        values.append(min(1.0, max(0.0, float(value))))
    if not values:
        return None
    return sum(values) / len(values)


def check(
    function: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    issues: Optional[Iterable[str]] = None,
) -> Any:
    """Decorator that names a check and can limit it to some issues.

        @check(issues=["missing_citation"])
        def cites_sources(case, output):
            return "[source]" in output
    """

    def wrap(fn: Callable) -> Callable:
        fn.__name__ = name or getattr(fn, "__name__", "check")
        fn.issues = tuple(issues) if issues else ()
        return fn

    return wrap(function) if function is not None else wrap


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str, sort_keys=True)


def terms(case: Case, output: Any) -> Optional[float]:
    """The share of the case's ``must`` terms present and ``must_not`` terms absent."""
    if not case.must and not case.must_not:
        return None
    text = _text(output).lower()
    hits = sum(str(term).lower() in text for term in case.must)
    hits += sum(str(term).lower() not in text for term in case.must_not)
    return hits / (len(case.must) + len(case.must_not))


def reference(scorer: Any = None) -> Check:
    """Similarity of the answer to ``case.expect``, the answer that was wanted.

    Text is compared with ``scorer`` (anything with ``score(reference,
    candidate)``; the default compares wording, not meaning). Other values
    are compared for equality.
    """
    scorer = scorer or HybridStringSimilarityScorer()

    def matches_reference(case: Case, output: Any) -> Optional[float]:
        if case.expect is None:
            return None
        if isinstance(case.expect, str) and isinstance(output, str):
            return float(scorer.score(case.expect, output))
        return 1.0 if output == case.expect else 0.0

    return matches_reference


def stays_close_to_approved(scorer: Any, *, enough: float = 0.8) -> Check:
    """For approved cases: the new answer must still mean what the approved
    one did. ``scorer`` is anything with ``score(reference, candidate)``;
    similarity of ``enough`` or more counts as full marks.

    Give it a scorer that compares meaning (an embedding scorer, say).
    A scorer that compares wording cannot do this job: on a 3.8B local
    model, adding one harmless instruction moved wording similarity to the
    approved answers from 0.88 to 0.26, and a harmful instruction moved it
    to 0.24. That is why this check is not on by default.
    """

    def close_to_approved(case: Case, output: Any) -> Optional[float]:
        if case.kind != APPROVED or not isinstance(case.output, str) or not isinstance(output, str):
            return None
        similarity = float(scorer.score(case.output, output))
        return 1.0 if similarity >= enough else similarity / enough

    return close_to_approved


def equals(case: Case, output: Any) -> Optional[float]:
    """Exact match with ``case.expect`` (text is compared ignoring case and spacing)."""
    if case.expect is None:
        return None
    if isinstance(case.expect, str) and isinstance(output, str):
        squash = lambda text: re.sub(r"\s+", " ", text).strip().lower()  # noqa: E731
        return 1.0 if squash(case.expect) == squash(output) else 0.0
    return 1.0 if output == case.expect else 0.0


def contains(*wanted: str) -> Check:
    """Every answer must mention all of these terms."""

    def contains_terms(case: Case, output: Any) -> float:
        text = _text(output).lower()
        return sum(term.lower() in text for term in wanted) / len(wanted)

    return contains_terms


def excludes(*unwanted: str) -> Check:
    """No answer may mention any of these terms."""

    def excludes_terms(case: Case, output: Any) -> float:
        text = _text(output).lower()
        return 0.0 if any(term.lower() in text for term in unwanted) else 1.0

    return excludes_terms


def matches(pattern: str, flags: int = re.IGNORECASE) -> Check:
    """The answer must match this regular expression somewhere."""
    compiled = re.compile(pattern, flags)

    def matches_pattern(case: Case, output: Any) -> float:
        return 1.0 if compiled.search(_text(output)) else 0.0

    return matches_pattern


def max_words(limit: int) -> Check:
    """The answer must be at most ``limit`` words long."""

    def within_length(case: Case, output: Any) -> float:
        return 1.0 if len(_text(output).split()) <= limit else 0.0

    return within_length


def valid_json(*required_keys: str) -> Check:
    """The answer must be JSON (already parsed, or text) with these keys."""

    def is_valid_json(case: Case, output: Any) -> float:
        value = output
        if isinstance(output, str):
            try:
                value = json.loads(output)
            except ValueError:
                return 0.0
        if not required_keys:
            return 1.0
        if not isinstance(value, dict):
            return 0.0
        return sum(key in value for key in required_keys) / len(required_keys)

    return is_valid_json


def judged(model: Callable[[str], str], rubric: str) -> Check:
    """Let a model grade the answer against a rubric.

    ``model`` is any function from a prompt to a reply. A grader is a
    hypothesis about quality, not a fact: use one only when it is clearly
    stronger than the system it grades. A reply with no readable score counts
    as "nothing to say", not as zero.
    """
    from erratica.learning.critic import parse_score

    def judged_by_model(case: Case, output: Any) -> Optional[float]:
        prompt = (
            "Grade the answer below against the rubric.\n"
            f"Rubric: {rubric}\n\n"
            f"Request: {_text(case.input)}\n\n"
            f"Answer: {_text(output)}\n\n"
        )
        if case.expect is not None:
            prompt += f"An answer that was wanted: {_text(case.expect)}\n\n"
        prompt += "Reply with one line: SCORE: n/10"
        return parse_score(model(prompt))

    return judged_by_model


def default_checks() -> List[Check]:
    """What is used when no checks are given: the case's own terms and reference."""
    return [terms, reference()]
