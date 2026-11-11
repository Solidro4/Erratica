"""Critics and self-review: finding mistakes without waiting for a human.

A critic grades an answer. Self-review uses one to look at interactions
nobody rated, flag the weak ones, and try to write a better answer.

What a critic produces is a hypothesis, not ground truth, and the rest of
Erratica treats it that way. Feedback written by self-review is stored with
``source='critic'`` and, unless you opt in:

- it is not recalled as a reference example at answer time
- it is not exported for fine-tuning
- it counts for half in triage's reference-strength channel

It can still drive rule and example proposals, and those still have to pass
replay, including the regression guard on answers real users approved. That
guard is what keeps a confidently wrong critic from steering the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


class Critic(Protocol):
    name: str

    def score(
        self,
        *,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
        route: str = "general",
    ) -> Optional[float]:
        """Quality of the response in [0, 1], or None if it cannot be judged."""
        ...


class ModelCritic:
    """Uses a language model as a grader.

    Works with any adapter that has ``generate(prompt, system_prompt,
    route)``. A critic is only useful if it is meaningfully stronger than
    the model it grades, so point this at the largest model you can run.
    """

    name = "model_critic"

    _SYSTEM = (
        "You are a strict grader of AI assistant answers. You judge whether an answer is "
        "correct, complete, and directly useful for the request."
    )

    def __init__(self, model: Any, *, rubric: Optional[str] = None) -> None:
        self.model = model
        self.rubric = rubric

    def score(
        self,
        *,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
        route: str = "general",
    ) -> Optional[float]:
        parts = [f"Request:\n{prompt}", f"Answer to grade:\n{response}"]
        if reference:
            parts.append(f"Reference answer known to be good:\n{reference}")
        if self.rubric:
            parts.append(f"Grading rubric:\n{self.rubric}")
        parts.append(
            "Grade the answer from 0 (useless or wrong) to 10 (could not be better). "
            "Reply with exactly one line in the form: SCORE: <number>"
        )
        raw = self.model.generate("\n\n".join(parts), self._SYSTEM, route)
        return parse_score(raw)

    def revise(self, *, prompt: str, response: str, route: str = "general") -> Optional[str]:
        """Write a better answer to the same request."""
        request = "\n\n".join(
            [
                f"Request:\n{prompt}",
                f"A weak answer that was given:\n{response}",
                "Write the answer that should have been given. Reply with the answer only.",
            ]
        )
        raw = self.model.generate(
            request,
            "You are an expert assistant. You write the best possible answer to a request.",
            route,
        )
        text = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.DOTALL).strip()
        return text or None


def parse_score(raw: Any) -> Optional[float]:
    text = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.DOTALL)
    match = re.search(r"SCORE\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/\s*10)?", text, flags=re.I)
    if not match:
        match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*/\s*10\b", text)
    if not match:
        return None
    value = float(match.group(1))
    if value > 10:
        return None
    return value / 10.0


@dataclass
class SelfReviewFinding:
    interaction_id: int
    score: float
    revised_score: Optional[float] = None
    correction_recorded: bool = False


@dataclass
class SelfReviewReport:
    reviewed: int = 0
    flagged: int = 0
    corrections: int = 0
    unjudged: int = 0
    findings: list[SelfReviewFinding] = field(default_factory=list)


class SelfReviewer:
    """Grades unrated interactions and records what the critic finds."""

    ISSUE_TYPE = "self_review"

    def __init__(self, db: Any, feedback: Any) -> None:
        self.db = db
        self.feedback = feedback

    def run(
        self,
        critic: Any,
        *,
        limit: int = 20,
        threshold: float = 0.6,
        margin: float = 0.2,
    ) -> SelfReviewReport:
        """Review up to ``limit`` unrated interactions.

        An answer scoring below ``threshold`` is flagged. If the critic can
        also write an answer that it grades at least ``margin`` higher, that
        answer is stored as a critic-sourced correction.
        """
        report = SelfReviewReport()
        for interaction in self.db.list_unreviewed_interactions(limit=limit):
            report.reviewed += 1
            score = critic.score(
                prompt=interaction["prompt"],
                response=interaction["response"],
                route=interaction["route"],
            )
            if score is None:
                report.unjudged += 1
                continue
            if score >= threshold:
                # Remember the pass so this answer is not graded again. It is
                # stored as a critic approval, which the regression guard
                # ignores: only answers a person approved are protected.
                self.feedback.record(
                    interaction["id"],
                    rating=1,
                    notes=f"critic '{getattr(critic, 'name', 'critic')}' scored this {score:.2f}",
                    source="critic",
                )
                continue

            report.flagged += 1
            finding = SelfReviewFinding(interaction_id=interaction["id"], score=score)
            candidate = None
            if hasattr(critic, "revise"):
                candidate = critic.revise(
                    prompt=interaction["prompt"],
                    response=interaction["response"],
                    route=interaction["route"],
                )
            if candidate:
                finding.revised_score = critic.score(
                    prompt=interaction["prompt"],
                    response=candidate,
                    route=interaction["route"],
                )
            verified = (
                candidate is not None
                and finding.revised_score is not None
                and finding.revised_score - score >= margin
            )
            self.feedback.record(
                interaction["id"],
                rating=-1,
                issue_type=self.ISSUE_TYPE,
                ideal_response=candidate if verified else None,
                notes=(
                    f"critic '{getattr(critic, 'name', 'critic')}' scored this {score:.2f}"
                    + (f"; its revision scored {finding.revised_score:.2f}" if verified else "")
                ),
                source="critic",
            )
            if verified:
                finding.correction_recorded = True
                report.corrections += 1
            report.findings.append(finding)
        return report
