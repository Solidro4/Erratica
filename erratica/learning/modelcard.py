from __future__ import annotations

from datetime import date
from pathlib import Path

from erratica.storage.db import Database

_BEHAVIOR_TYPES = ("prompt_rule", "few_shot", "tool")


class ModelCardBuilder:
    """Generates a model card from an Erratica learning database.

    The card is built entirely from the audit trail, so anyone holding the
    database can check every number on it. It states how each change was
    validated, including the weak cases: a patch judged in-sample is listed
    as such rather than folded in with the held-out ones.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def build(self, *, model_name: str = "local model") -> str:
        metrics = self.db.get_dashboard_snapshot()["metrics"]
        patches = list(reversed(self.db.list_patch_history(limit=100000)))
        triage = self.db.list_triage(limit=100000)
        human_corrections = self.db.list_corrections(limit=100000)
        all_corrections = self.db.list_corrections(limit=100000, sources=None)

        promoted = [patch for patch in patches if patch["status"] == "promoted"]
        rejected = [patch for patch in patches if patch["status"] == "rejected"]
        rolled_back = [patch for patch in patches if patch["status"] == "rolled_back"]
        quarantined = {
            (entry.get("route"), entry["issue_type"])
            for entry in triage
            if entry["verdict"] == "quarantined"
        }
        live_behavior = [patch for patch in promoted if patch["patch_type"] in _BEHAVIOR_TYPES]
        held_out = [
            patch for patch in live_behavior if patch["details"].get("validation") == "holdout"
        ]
        weight_cycles = [patch for patch in patches if patch["patch_type"] == "model_weights"]

        lines = [
            f"# Model Card: {model_name}",
            "",
            f"Improved with [Erratica](https://github.com/Solidro4/Erratica). "
            f"Generated on {date.today().isoformat()}.",
            "",
            "Every behavior change listed here was replayed against recorded failures",
            "and promoted only if it improved them without making approved answers",
            "worse. Where there were enough cases, the change was judged on failures",
            "it was not written from (held-out validation).",
            "",
            "## Learning Provenance",
            "",
            f"- Interactions observed: {metrics['interactions']}",
            f"- Feedback items: {metrics['feedback_items']} "
            f"({metrics['approvals']} approvals, {metrics['self_review_items']} from self-review)",
            f"- Behavior updates promoted: {len(promoted)}",
            f"- Behavior updates rejected: {len(rejected)}",
            f"- Behavior updates rolled back: {len(rolled_back)}",
            f"- Mistake clusters quarantined as noise: {len(quarantined)}",
            f"- Human corrections available for fine-tuning: {len(human_corrections)}",
            f"- Critic-written corrections (excluded from training by default): "
            f"{len(all_corrections) - len(human_corrections)}",
        ]
        if promoted or rejected:
            total = len(promoted) + len(rejected) + len(rolled_back)
            lines.append(
                f"- Validation pass rate: {len(promoted) + len(rolled_back)}/{total} proposals survived replay"
            )
        if live_behavior:
            lines.append(
                f"- Live patches validated on held-out cases: {len(held_out)}/{len(live_behavior)}"
            )

        lines += ["", "## Learned Behavior", ""]
        if live_behavior:
            for patch in live_behavior:
                lines.append(f"- {self._summarise(patch)}")
        else:
            lines.append("- No live behavior updates.")

        if weight_cycles:
            lines += ["", "## Weight Cycles", ""]
            for patch in weight_cycles:
                details = patch["details"]
                lines.append(f"- {self._summarise(patch)}")
                if details.get("train_score_after") is not None:
                    lines.append(
                        f"  - on its own training prompts: "
                        f"{details.get('train_score_before', 0.0):.2f} -> {details['train_score_after']:.2f}"
                    )

        lines += ["", "## Rejected Updates", ""]
        if rejected:
            for patch in rejected[-10:]:
                reason = patch["details"].get("reason", "")
                lines.append(
                    f"- {patch['patch_type']}: {self._describe(patch)}"
                    + (f" ({reason})" if reason else "")
                )
        else:
            lines.append("- No rejected updates recorded yet.")

        if rolled_back:
            lines += ["", "## Rolled Back", ""]
            for patch in rolled_back:
                reason = patch["details"].get("rollback_reason", "")
                lines.append(
                    f"- {patch['patch_type']}: {self._describe(patch)}"
                    + (f" ({reason})" if reason else "")
                )

        lines += [
            "",
            "## How This Card Was Produced",
            "",
            "All numbers come from the Erratica audit trail (SQLite). To reproduce:",
            "",
            "```bash",
            "erratica modelcard --db data/erratica.db",
            "```",
            "",
        ]
        return "\n".join(lines)

    def _summarise(self, patch: dict) -> str:
        details = patch["details"]
        parts = [f"**{patch['patch_type']}**"]
        if patch.get("route"):
            parts.append(f"[{patch['route']}]")
        if patch["score_before"] is not None and patch["score_after"] is not None:
            parts.append(f"score {patch['score_before']:.2f} -> {patch['score_after']:.2f}")
        validation = details.get("validation")
        if validation:
            label = {
                "holdout": "held-out",
                "in_sample": "in-sample, weak evidence",
                "manual": "hand-written, judged on all recorded failures",
            }.get(validation, validation)
            parts.append(f"on {details.get('cases_scored', '?')} case(s), {label}")
        status = "" if patch["status"] == "promoted" else f" ({patch['status']})"
        return " ".join(parts) + f": {self._describe(patch)}{status}"

    def _describe(self, patch: dict) -> str:
        content = patch["content"]
        return (
            content.get("rule_text")
            or content.get("description")
            or f"Reference example for route '{content.get('route', 'general')}'"
        )

    def write(self, path: str | Path, *, model_name: str = "local model") -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.build(model_name=model_name), encoding="utf-8")
        return output_path
