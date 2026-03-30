from __future__ import annotations

from synora.policy.examples import ExampleStore
from synora.storage.db import Database

_BASE_RULES = [
    "You are a private local AI runtime.",
    "Be concise, grounded, and explicit about missing information.",
    "Follow active learned rules before finalizing the answer.",
]


class PromptPolicy:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.examples = ExampleStore(db)

    def version(self) -> int:
        return self.db.get_policy_version()

    def active_rules(self) -> list[str]:
        rows = self.db.list_policy_rules(active_only=True)
        return [row["rule_text"] for row in rows]

    def render_system_prompt(
        self,
        *,
        route: str,
        extra_rules: list[str] | None = None,
    ) -> str:
        lines = list(_BASE_RULES)
        lines.append(f"Current route: {route}.")

        rules = self.active_rules()
        if extra_rules:
            rules.extend(extra_rules)
        if rules:
            lines.append("Active learned rules:")
            for index, rule in enumerate(rules, start=1):
                lines.append(f"{index}. {rule}")

        rendered_examples = self.examples.render(route=route, limit=2)
        if rendered_examples:
            lines.append("Reference examples:")
            lines.append(rendered_examples)

        return "\n".join(lines)

    def apply_prompt_rule(self, rule_text: str, *, source_patch_id: int | None = None) -> int:
        return self.db.insert_policy_rule(
            rule_text=rule_text,
            source_patch_id=source_patch_id,
            active=True,
        )
