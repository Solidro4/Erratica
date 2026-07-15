from __future__ import annotations

from datetime import date
from pathlib import Path

from synora.storage.db import Database


class ModelCardBuilder:
    """Generates a verifiable model card from a Synora learning database.

    Every model improved with Synora can be published together with a card
    that documents what it learned, how each change was validated, and what
    was rejected or quarantined. The card is built entirely from the audit
    trail, so the claims on it are reproducible from the database.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def build(self, *, model_name: str = "local model") -> str:
        snapshot = self.db.get_dashboard_snapshot()
        metrics = snapshot["metrics"]
        patches = self.db.list_patch_history(limit=1000)
        triage = self.db.list_triage(limit=1000)
        corrections = self.db.list_corrections(limit=1000)

        promoted = [patch for patch in patches if patch["status"] == "promoted"]
        rejected = [patch for patch in patches if patch["status"] == "rejected"]
        quarantined = [entry for entry in triage if entry["verdict"] == "quarantined"]

        lines = [
            f"# Model Card: {model_name}",
            "",
            f"Improved with [Synora](https://github.com/Solidro4/Synora) - generated on {date.today().isoformat()}.",
            "",
            "This model does not just ship as-is: it carries validated experience",
            "learned from real usage. Every behavior change below was replayed",
            "against past failures and promoted only when it measurably improved",
            "results without regressing previous wins.",
            "",
            "## Learning Provenance",
            "",
            f"- Interactions observed: {metrics['interactions']}",
            f"- Feedback items received: {metrics['feedback_items']}",
            f"- Behavior updates promoted: {len(promoted)}",
            f"- Behavior updates rejected: {len(rejected)}",
            f"- Mistake clusters quarantined as noise: {len(quarantined)}",
            f"- User corrections available for fine-tuning: {len(corrections)}",
        ]

        if promoted or rejected:
            total = len(promoted) + len(rejected)
            lines.append(f"- Validation pass rate: {len(promoted)}/{total} proposals survived replay")

        lines.extend(["", "## Learned Behavior", ""])
        if promoted:
            for patch in promoted:
                gain = ""
                if patch["score_before"] is not None and patch["score_after"] is not None:
                    gain = f" (score {patch['score_before']:.2f} -> {patch['score_after']:.2f})"
                lines.append(f"- **{patch['patch_type']}**{gain}: {self._describe(patch)}")
        else:
            lines.append("- No promoted behavior updates yet.")

        lines.extend(["", "## Rejected Updates (proof of selectivity)", ""])
        if rejected:
            for patch in rejected[:10]:
                lines.append(f"- {patch['patch_type']}: {self._describe(patch)}")
        else:
            lines.append("- No rejected updates recorded yet.")

        lines.extend(
            [
                "",
                "## How This Card Was Produced",
                "",
                "All numbers come from the Synora audit trail (SQLite). To reproduce:",
                "",
                "```bash",
                "python -m synora.cli.modelcard --db data/synora.db",
                "```",
                "",
            ]
        )
        return "\n".join(lines)

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
