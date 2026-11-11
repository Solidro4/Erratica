from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from erratica.domains import DomainRegistry
from erratica.learning.clustering import FailureCluster

_FORMAT_PHRASES = {
    "bullet_list": "Format the answer as a bullet list.",
    "numbered_list": "Format the answer as a numbered list.",
    "sections": "Organise the answer into labelled sections.",
    "paragraph": "Answer in a single paragraph, without lists.",
    "json": "Answer with valid JSON only.",
    "code_block": "Put code in a fenced code block.",
}


@dataclass
class PatchProposal:
    patch_type: str
    target: str
    rule_text: str
    rationale: str
    source_issue_type: str
    route: Optional[str] = None
    origin: str = "template"
    example_prompt: Optional[str] = None
    example_response: Optional[str] = None
    tool_name: Optional[str] = None
    tool_source: Optional[str] = None

    def content(self) -> dict[str, Any]:
        if self.patch_type == "few_shot":
            return {
                "route": self.route,
                "prompt": self.example_prompt,
                "response": self.example_response,
            }
        if self.patch_type == "tool":
            return {
                "name": self.tool_name,
                "source": self.tool_source,
                "description": self.rule_text,
            }
        return {"rule_text": self.rule_text, "origin": self.origin}

    def as_record(self) -> dict[str, Any]:
        return {
            "patch_type": self.patch_type,
            "target": self.target,
            "rule_text": self.rule_text,
            "rationale": self.rationale,
            "source_issue_type": self.source_issue_type,
            "route": self.route,
        }


class PromptRulePatcher:
    """First rung: a rule from the domain pack or from the feedback signals.

    Costs nothing to propose. Returns None when neither the domain pack nor
    the feedback says anything concrete enough to turn into an instruction,
    instead of inventing a filler rule.
    """

    def __init__(self, registry: Optional[DomainRegistry] = None) -> None:
        self.registry = registry or DomainRegistry()

    def propose(self, cluster: FailureCluster) -> Optional[PatchProposal]:
        rule_text = self._rule_for_cluster(cluster)
        if not rule_text:
            return None
        return PatchProposal(
            patch_type="prompt_rule",
            target="system_prompt",
            rule_text=rule_text,
            rationale=(
                f"{len(cluster.failing_cases)} failure(s) tagged '{cluster.issue_type}' "
                f"on route '{cluster.route}'."
            ),
            source_issue_type=cluster.issue_type,
            route=cluster.route,
            origin="template",
        )

    def _rule_for_cluster(self, cluster: FailureCluster) -> str:
        parts: list[str] = []
        template = self.registry.rule_template(cluster.route, cluster.issue_type)
        if template:
            parts.append(template)
        elif cluster.required_terms:
            parts.append(
                "Make sure the answer covers these required terms when the prompt supports them: "
                + ", ".join(cluster.required_terms)
                + "."
            )

        common_format = cluster.preferred_formats[0] if cluster.preferred_formats else None
        phrase = _FORMAT_PHRASES.get(common_format or "")
        if phrase:
            parts.append(phrase)
        return " ".join(parts)


class ModelRulePatcher:
    """Second rung: ask a proposer model to write the rule.

    The proposer reads real failures with their feedback and writes one
    general instruction. It can be the answering model itself or a larger
    local model. Whatever it writes is only a candidate: it still has to win
    on held-out cases like any other patch.
    """

    _SYSTEM = (
        "You improve an AI assistant by writing one instruction for its system prompt. "
        "You are shown requests it answered badly and what was wrong."
    )

    def __init__(self, proposer: Any, *, max_cases: int = 4) -> None:
        self.proposer = proposer
        self.max_cases = max_cases

    def propose(self, cluster: FailureCluster) -> Optional[PatchProposal]:
        cases = cluster.failing_cases[: self.max_cases]
        if not cases:
            return None
        request = "\n\n".join(
            [
                f"Task type: {cluster.route}. Reported problem: {cluster.issue_type.replace('_', ' ')}.",
                describe_cases(cases),
                "Write ONE instruction that would have prevented these failures and will also "
                "help on similar future requests. Rules: imperative voice, at most 40 words, "
                "general (no names, numbers, or IDs copied from the examples). "
                "Reply with the instruction only.",
            ]
        )
        raw = self.proposer.generate(request, self._SYSTEM, cluster.route)
        rule_text = clean_instruction(raw)
        if not rule_text:
            return None
        return PatchProposal(
            patch_type="prompt_rule",
            target="system_prompt",
            rule_text=rule_text,
            rationale=(
                f"Written by the proposer model from {len(cases)} failure(s) tagged "
                f"'{cluster.issue_type}' on route '{cluster.route}'."
            ),
            source_issue_type=cluster.issue_type,
            route=cluster.route,
            origin="model",
        )


class FewShotPatcher:
    """Third rung: teach by example when a rule is not enough.

    Proposes one reference answer from the cluster as an example for the
    route. Replay never shows an example to the case it came from, so it
    can only win by helping on other prompts.
    """

    def propose(self, cluster: FailureCluster) -> Optional[PatchProposal]:
        best_case = self._best_reference_case(cluster)
        if best_case is None:
            return None

        route = str(best_case.get("route") or cluster.route or "general")
        reference = str(best_case.get("ideal_response") or best_case.get("correction"))
        return PatchProposal(
            patch_type="few_shot",
            target="few_shot_examples",
            rule_text=f"Add a reference example for route '{route}'.",
            rationale=(
                f"Cluster '{cluster.issue_type}' includes a reference answer; "
                "teach the desired behavior by example."
            ),
            source_issue_type=cluster.issue_type,
            route=route,
            origin=str(best_case.get("source") or "human"),
            example_prompt=str(best_case.get("prompt")),
            example_response=reference,
        )

    def _best_reference_case(self, cluster: FailureCluster) -> Optional[dict[str, Any]]:
        candidates = [
            case
            for case in cluster.failing_cases
            if case.get("ideal_response") or case.get("correction")
        ]
        if not candidates:
            return None
        # Prefer answers a human wrote, then the most complete one.
        return max(
            candidates,
            key=lambda case: (
                (case.get("source") or "human") == "human",
                len(str(case.get("ideal_response") or case.get("correction"))),
            ),
        )


class ToolPatcher:
    """Fourth rung: ask the proposer to write a small Python tool.

    For failures that are really computation (arithmetic, parsing, date
    maths), no instruction makes a small model reliable. A tool computes the
    answer and hands it to the model as a verified note. The source is
    screened and sandboxed (see ``erratica.tools.sandbox``) and the tool is
    promoted only if replay with it beats replay without it.
    """

    _SYSTEM = (
        "You write small, safe Python helper functions that compute facts an AI assistant "
        "keeps getting wrong."
    )

    def __init__(self, proposer: Any, *, max_cases: int = 3) -> None:
        self.proposer = proposer
        self.max_cases = max_cases

    def propose(self, cluster: FailureCluster) -> Optional[PatchProposal]:
        request = self._request(cluster)
        if request is None:
            return None
        return self._proposal(cluster, self.proposer.generate(request, self._SYSTEM, cluster.route))

    def repair(
        self,
        cluster: FailureCluster,
        previous: PatchProposal,
        observations: str,
    ) -> Optional[PatchProposal]:
        """Show the proposer what its tool actually did and ask for a fix."""
        request = self._request(cluster)
        if request is None:
            return None
        request = "\n\n".join(
            [
                request,
                f"Your previous attempt:\n```python\n{previous.tool_source}\n```",
                f"What happened when it was tried:\n{observations}",
                "Find the bug and write a corrected function. "
                "Reply with one Python code block and nothing else.",
            ]
        )
        proposal = self._proposal(cluster, self.proposer.generate(request, self._SYSTEM, cluster.route))
        if proposal is not None:
            proposal.rationale += " Revised after a failed trial on the training cases."
        return proposal

    def _request(self, cluster: FailureCluster) -> Optional[str]:
        cases = cluster.failing_cases[: self.max_cases]
        if not cases:
            return None
        return "\n\n".join(
            [
                f"Task type: {cluster.route}. Reported problem: {cluster.issue_type.replace('_', ' ')}.",
                describe_cases(cases),
                "Write a Python function with exactly this signature:\n\n"
                "def run(prompt: str) -> str:\n\n"
                "It receives the user's request text and returns a short note (one or two "
                "lines) with the computed facts needed to answer correctly, or an empty "
                "string when it does not apply. It must work for similar requests with "
                "different values, so parse the values from the prompt instead of hardcoding "
                "them. You may import only: math, re, json, statistics, datetime, decimal, "
                "fractions, itertools, collections, string, calendar. No file, network, or "
                "system access. Reply with one Python code block and nothing else.",
            ]
        )

    def _proposal(self, cluster: FailureCluster, raw: Any) -> Optional[PatchProposal]:
        source = extract_code(raw)
        if not source:
            return None
        name = re.sub(r"[^a-z0-9_]", "_", f"{cluster.route}_{cluster.issue_type}".lower())
        return PatchProposal(
            patch_type="tool",
            target="tools",
            rule_text=f"Add a computed-note tool for '{cluster.issue_type}' on route '{cluster.route}'.",
            rationale=(
                f"Written by the proposer model from {len(cluster.failing_cases[: self.max_cases])} "
                "failure(s) that rules and examples did not fix."
            ),
            source_issue_type=cluster.issue_type,
            route=cluster.route,
            origin="model",
            tool_name=name,
            tool_source=source,
        )


# ----------------------------------------------------------------------
# Helpers shared by the model-driven patchers
# ----------------------------------------------------------------------


def _clip(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def describe_cases(cases: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, case in enumerate(cases, start=1):
        lines = [
            f"Failure {index}",
            f"Request: {_clip(case.get('prompt'), 600)}",
            f"Bad answer: {_clip(case.get('response'), 400)}",
        ]
        if case.get("notes"):
            lines.append(f"Feedback: {_clip(case.get('notes'), 300)}")
        if case.get("required_terms"):
            lines.append("Should mention: " + ", ".join(case["required_terms"]))
        if case.get("preferred_format"):
            lines.append(f"Wanted format: {case['preferred_format']}")
        reference = case.get("ideal_response") or case.get("correction")
        if reference:
            lines.append(f"Wanted answer: {_clip(reference, 600)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def clean_instruction(raw: Any, *, max_chars: int = 400) -> str:
    """Reduce a model reply to a single instruction line."""
    text = str(raw or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for line in text.splitlines():
        candidate = line.strip().strip("`*\"' ")
        candidate = re.sub(r"^(?:[-*•]|\d+[.)]|(?:instruction|rule)\s*:)\s*", "", candidate, flags=re.I)
        if len(candidate) >= 12:
            return candidate[:max_chars]
    return ""


def extract_code(raw: Any) -> str:
    """Pull the Python source out of a model reply."""
    text = str(raw or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    for block in fenced:
        if "def run" in block:
            return block.strip()
    if "def run" in text and "```" not in text:
        return text.strip()
    return ""
