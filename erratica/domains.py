"""Domain packs: everything Erratica knows about one kind of task.

The learning loop itself is domain-agnostic. A domain pack tells it how to
recognise prompts of one kind, which rule to try first for a known issue
type, and how to check an answer beyond comparing it with a reference.

Register your own with ``Erratica(domains=[...])``; the built-in packs are
small examples, not the limit of what the loop can cover.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# A check looks at one replayed case and the model's answer and returns a
# score in [0, 1], or None when it has nothing to say about this case.
Check = Callable[[Any, str], Optional[float]]

_NUMBER = r"\$?[0-9]+(?:\.[0-9]+)?[MKmk%]?"
_LIST_LINE = r"(?m)^\s*(?:[-*•]|\d+[.)])\s+"


@dataclass(frozen=True)
class Domain:
    name: str
    keywords: tuple[str, ...] = ()
    description: str = ""
    # issue type -> rule to try first when that issue recurs
    rule_templates: dict[str, str] = field(default_factory=dict)
    # issue type -> check applied to cases with that issue type
    issue_checks: dict[str, Check] = field(default_factory=dict)
    # checks applied to every case on this route
    checks: tuple[Check, ...] = ()
    # per-route promotion gates; None means use the engine defaults
    min_delta: Optional[float] = None
    regression_tolerance: Optional[float] = None


# ----------------------------------------------------------------------
# Checks that make sense for any domain
# ----------------------------------------------------------------------


def has_list(response: str) -> bool:
    return bool(re.search(_LIST_LINE, response))


def check_structure(case: Any, response: str) -> float:
    return 1.0 if has_list(response) else 0.0


def check_numbers_grounded(case: Any, response: str) -> float:
    """Every number in the answer must appear in the prompt."""
    prompt_numbers = {match.lower() for match in re.findall(_NUMBER, case.prompt)}
    response_numbers = {match.lower() for match in re.findall(_NUMBER, response)}
    if not response_numbers:
        return 1.0
    grounded = sum(number in prompt_numbers for number in response_numbers)
    return grounded / len(response_numbers)


GENERIC_ISSUE_CHECKS: dict[str, Check] = {
    "missing_structure": check_structure,
    "hallucinated_values": check_numbers_grounded,
}

GENERIC_RULE_TEMPLATES: dict[str, str] = {
    "missing_structure": "Use structured output with clear headings or ordered points.",
    "hallucinated_values": (
        "Do not invent facts or numbers. If a value is not given in the prompt, say it is unavailable."
    ),
    "too_long": "Answer in as few sentences as the question allows. Do not add preamble or recap.",
    "too_vague": "Be specific: name the concrete items, values, or steps the prompt asks about.",
}


# ----------------------------------------------------------------------
# Support pack
# ----------------------------------------------------------------------

_RESOLUTION_WORDS = (
    "resolution",
    "refund",
    "revers",
    "replac",
    "escalat",
    "credit",
    "reship",
    "cancel",
    "stopping renewal",
)
_TIMEFRAME = r"\b(?:\d+\s*-\s*\d+|\d+)\s+(?:business\s+days?|days?|hours?|weeks?)\b|\bwithin\s+\d+"


def _has_resolution(text: str) -> bool:
    return any(word in text for word in _RESOLUTION_WORDS)


def _has_timeframe(text: str) -> bool:
    return bool(re.search(_TIMEFRAME, text))


def _support_subjects(prompt: str) -> list[str]:
    lowered = prompt.lower()
    subjects = []
    for word in ("refund", "order", "charge", "billing", "shipment", "replacement", "subscription"):
        if word in lowered:
            subjects.append(word)
    return subjects[:4]


def check_support_reply(case: Any, response: str) -> float:
    """A useful support reply names the subject, commits to an action, and gives a timeframe."""
    lowered = response.lower()
    parts = [
        1.0 if _has_resolution(lowered) else 0.0,
        1.0 if _has_timeframe(lowered) else 0.0,
    ]
    subjects = _support_subjects(case.prompt)
    if subjects:
        parts.append(sum(subject in lowered for subject in subjects) / len(subjects))
    return sum(parts) / len(parts)


def _check_resolution(case: Any, response: str) -> float:
    lowered = response.lower()
    return 1.0 if _has_resolution(lowered) and _has_timeframe(lowered) else 0.0


def _check_next_steps(case: Any, response: str) -> float:
    lowered = response.lower()
    return 1.0 if "next step" in lowered or has_list(response) else 0.0


def _check_policy(case: Any, response: str) -> float:
    lowered = response.lower()
    return 1.0 if "policy" in lowered or _has_timeframe(lowered) else 0.0


SUPPORT = Domain(
    name="support",
    keywords=("ticket", "refund", "customer", "support", "billing", "order", "shipment", "subscription"),
    description="Replies to customer support requests.",
    rule_templates={
        "missing_resolution": (
            "For support replies, restate the customer's issue, provide a concrete resolution, "
            "and include a follow-up window."
        ),
        "missing_next_steps": (
            "When answering support-style requests, always include concrete next steps "
            "and an expected follow-up window."
        ),
        "missing_policy": (
            "When refund or billing policy is relevant, include the applicable policy or payout timeline."
        ),
        "missing_entities": (
            "For support replies, reference the customer's order or account details "
            "and avoid vague acknowledgements."
        ),
        "too_vague": (
            "For support replies, reference the customer's order or account details "
            "and avoid vague acknowledgements."
        ),
    },
    issue_checks={
        "missing_resolution": _check_resolution,
        "missing_next_steps": _check_next_steps,
        "missing_policy": _check_policy,
    },
    checks=(check_support_reply,),
)


# ----------------------------------------------------------------------
# Finance pack
# ----------------------------------------------------------------------


def _check_segmentation(case: Any, response: str) -> float:
    entries = re.findall(r"(?m)^\s*(?:\d+[.)]\s+|[-*]\s+)?[A-Za-z][A-Za-z\s\-]+:", response)
    return 1.0 if len(entries) >= 2 else 0.0


def _numbers(text: str) -> list[float]:
    values = []
    for match in re.findall(r"[0-9][0-9,]*(?:\.[0-9]+)?", text):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def check_total(case: Any, response: str) -> Optional[float]:
    """When the prompt lists labelled amounts, the answer must state their sum."""
    amounts = [
        float(value.replace(",", ""))
        for value in re.findall(r":\s*\$?([0-9][0-9,]*(?:\.[0-9]+)?)", case.prompt)
        if value.replace(",", "")
    ]
    if len(amounts) < 2:
        return None
    total = sum(amounts)
    return 1.0 if any(abs(number - total) < 0.005 for number in _numbers(response)) else 0.0


FINANCE = Domain(
    name="finance",
    keywords=("revenue", "quarter", "segment", "finance", "earnings", "margin", "forecast"),
    description="Summaries of financial and business figures.",
    rule_templates={
        "missing_segmentation": (
            "When the prompt includes segmented business data, explicitly list each segment and its value."
        ),
    },
    issue_checks={
        "missing_segmentation": _check_segmentation,
        "wrong_total": check_total,
    },
)

GENERAL = Domain(name="general", description="Anything no other domain claims.")

DEFAULT_DOMAINS: tuple[Domain, ...] = (SUPPORT, FINANCE, GENERAL)


class DomainRegistry:
    def __init__(self, domains: Optional[list[Domain]] = None) -> None:
        self._domains: dict[str, Domain] = {}
        for domain in domains if domains is not None else DEFAULT_DOMAINS:
            self.register(domain)
        if "general" not in self._domains:
            self.register(GENERAL)

    def register(self, domain: Domain) -> None:
        self._domains[domain.name] = domain

    def get(self, name: str) -> Domain:
        return self._domains.get(name) or Domain(name=name)

    def names(self) -> list[str]:
        return list(self._domains)

    def all(self) -> list[Domain]:
        return list(self._domains.values())

    def rule_template(self, route: str, issue_type: str) -> Optional[str]:
        return self.get(route).rule_templates.get(issue_type) or GENERIC_RULE_TEMPLATES.get(issue_type)

    def checks_for(self, route: str, issue_type: str) -> list[Check]:
        domain = self.get(route)
        checks: list[Check] = []
        issue_check = domain.issue_checks.get(issue_type) or GENERIC_ISSUE_CHECKS.get(issue_type)
        if issue_check is not None:
            checks.append(issue_check)
        checks.extend(domain.checks)
        return checks
