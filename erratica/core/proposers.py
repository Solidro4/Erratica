"""Proposers: where candidate changes come from.

A proposer is any callable ``proposer(context) -> changes``. It may return
one ``Change``, several, or nothing. The loop calls its proposers in the
order given and stops at the first change the gate accepts, so list the
cheapest idea first.

A proposer only ever sees the failures on the *training* side of the split.
The gate judges on the others, so a change cannot win by memorising the cases
it was written from.

The ones here cover the usual ladder (a written instruction, a worked
example, a different setting). Writing your own is a five-line function:

    def shorter_context(context):
        if context.issue == "timeout":
            yield Change.set("retriever.top_k", 3, kind="setting")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

from erratica.core.types import HOLDOUT, Case, Change
from erratica.learning.patcher import clean_instruction


@dataclass
class Context:
    """What a proposer is told."""

    issue: str
    failures: List[Case]
    config: Any
    # "holdout" when the gate will judge on failures this proposer cannot see.
    validation: str
    # Names of changes already rejected for this issue in this pass.
    rejected: List[str] = field(default_factory=list)


Proposer = Callable[[Context], Any]


def as_changes(result: Any) -> Iterator[Change]:
    """Normalise whatever a proposer returned."""
    if result is None:
        return
    if isinstance(result, Change):
        yield result
        return
    for change in result:
        if change is not None:
            yield change


def _clip(value: Any, limit: int) -> str:
    text = str(value if value is not None else "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def rules_from_feedback(
    templates: Optional[Dict[str, str]] = None,
    *,
    path: str = "rules",
) -> Proposer:
    """An instruction taken from what people said, with no model call.

    ``templates`` maps an issue to the instruction to try for it. Without a
    template, the instruction is built from the ``must`` and ``must_not``
    terms recorded on the failures.
    """
    templates = dict(templates or {})

    def propose(context: Context) -> Iterator[Change]:
        text = templates.get(context.issue)
        if not text:
            wanted = sorted({term for case in context.failures for term in case.must})
            unwanted = sorted({term for case in context.failures for term in case.must_not})
            parts = []
            if wanted:
                parts.append("Always mention: " + ", ".join(wanted) + ".")
            if unwanted:
                parts.append("Never mention: " + ", ".join(unwanted) + ".")
            text = " ".join(parts)
        if text:
            yield Change.append(path, text, name=f"rule: {_clip(text, 60)}", kind="rule", issue=context.issue)

    return propose


def rule_from_model(
    model: Callable[[str], str],
    *,
    path: str = "rules",
    max_cases: int = 4,
) -> Proposer:
    """Let a model read the failures and write one general instruction.

    ``model`` is any function from a prompt to a reply. What it writes is a
    hypothesis; it is installed only if the gate accepts it.
    """

    def propose(context: Context) -> Iterator[Change]:
        blocks = []
        for index, case in enumerate(context.failures[:max_cases], start=1):
            lines = [
                f"Failure {index}",
                f"Request: {_clip(case.input, 600)}",
                f"Bad answer: {_clip(case.output, 400)}",
            ]
            if case.note:
                lines.append(f"Feedback: {_clip(case.note, 300)}")
            if case.must:
                lines.append("Should mention: " + ", ".join(case.must))
            if case.must_not:
                lines.append("Should not mention: " + ", ".join(case.must_not))
            if case.expect is not None:
                lines.append(f"Wanted answer: {_clip(case.expect, 600)}")
            blocks.append("\n".join(lines))
        if not blocks:
            return
        prompt = (
            "An assistant keeps making the same kind of mistake "
            f"(labelled '{context.issue}'). Here are examples.\n\n"
            + "\n\n".join(blocks)
            + "\n\nWrite ONE general instruction, in one sentence, that would prevent this "
            "kind of mistake on new requests. Do not mention the specific examples. "
            "Reply with the instruction only."
        )
        text = clean_instruction(model(prompt))
        if text and f"rule: {_clip(text, 60)}" not in context.rejected:
            yield Change.append(path, text, name=f"rule: {_clip(text, 60)}", kind="rule", issue=context.issue)

    return propose


def examples_from_failures(*, path: str = "examples", max_chars: int = 2000) -> Proposer:
    """A worked example: a failed request with the answer that was wanted.

    Only offered when the gate can judge it on other failures. An example
    judged on its own case would simply be copied back, which proves nothing.
    """

    def propose(context: Context) -> Iterator[Change]:
        if context.validation != HOLDOUT:
            return
        for case in reversed(context.failures):  # most recent first
            if isinstance(case.expect, str) and 0 < len(case.expect) <= max_chars:
                example = {"input": case.input, "output": case.expect}
                yield Change.append(
                    path,
                    example,
                    name=f"example: {_clip(case.input, 60)}",
                    kind="example",
                    issue=context.issue,
                )
                return

    return propose


def try_values(path: str, values: Sequence[Any], *, issues: Optional[Iterable[str]] = None) -> Proposer:
    """Try other values for one setting (a model name, a temperature, a
    retrieval depth), in the order given."""
    only = set(issues) if issues else None

    def propose(context: Context) -> Iterator[Change]:
        if only is not None and context.issue not in only:
            return
        for value in values:
            yield Change.set(path, value, name=f"{path} = {value!r}", kind="setting", issue=context.issue)

    return propose
