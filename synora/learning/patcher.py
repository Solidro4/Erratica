from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from synora.learning.clustering import FailureCluster


@dataclass
class PatchProposal:
    patch_type: str
    target: str
    rule_text: str
    rationale: str
    source_issue_type: str

    def as_record(self) -> dict[str, Any]:
        return {
            "patch_type": self.patch_type,
            "target": self.target,
            "rule_text": self.rule_text,
            "rationale": self.rationale,
            "source_issue_type": self.source_issue_type,
        }


class PromptRulePatcher:
    def propose(self, cluster: FailureCluster) -> PatchProposal:
        issue_type = cluster.issue_type
        rule_text = self._rule_for_cluster(cluster)
        rationale = f"Observed {cluster.size} recurring failures tagged as '{issue_type}'."
        return PatchProposal(
            patch_type="prompt_rule",
            target="system_prompt",
            rule_text=rule_text,
            rationale=rationale,
            source_issue_type=issue_type,
        )

    def _rule_for_cluster(self, cluster: FailureCluster) -> str:
        issue_type = cluster.issue_type
        common_format = cluster.preferred_formats[0] if cluster.preferred_formats else None

        if issue_type == "missing_segmentation":
            rule = "When the prompt includes segmented business data, explicitly list each segment and its value."
        elif issue_type == "missing_resolution":
            rule = "For support replies, restate the customer's issue, provide a concrete resolution, and include a follow-up window."
        elif issue_type == "missing_next_steps":
            rule = "When answering support-style requests, always include concrete next steps and an expected follow-up window."
        elif issue_type == "missing_policy":
            rule = "When refund or billing policy is relevant, include the applicable policy or payout timeline."
        elif issue_type in ("missing_entities", "too_vague"):
            rule = "For support replies, reference the customer's order or account details and avoid vague acknowledgements."
        elif issue_type == "hallucinated_values":
            rule = "Do not invent facts or numbers. If a value is not grounded in the prompt, say it is unavailable."
        elif issue_type == "missing_structure":
            rule = "Use structured output with clear headings or ordered points when the task is a summary or recommendation."
        elif cluster.required_terms:
            joined_terms = ", ".join(cluster.required_terms)
            rule = f"Preserve these required terms when supported by the prompt: {joined_terms}."
        else:
            rule = f"Address recurring '{issue_type}' failures before returning the final answer."

        if common_format == "bullet_list" and "bullet" not in rule.lower():
            rule += " Format the answer as a bullet list."
        if common_format == "numbered_list" and "numbered" not in rule.lower():
            rule += " Use a numbered list."

        return rule
