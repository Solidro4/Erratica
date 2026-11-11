from __future__ import annotations

from typing import Any, Collection, Optional

from erratica.domains import DomainRegistry
from erratica.policy.examples import ExampleStore
from erratica.storage.db import Database
from erratica.tools.sandbox import render_tool_notes, run_tools

_BASE_RULES = [
    "Be concise, grounded, and explicit about missing information.",
    "Follow the learned rules below before finalizing the answer.",
]


class PromptPolicy:
    """The live behavior of the model: learned rules, examples, and tools.

    Everything here is scoped to a route, so a rule learned for support
    replies never appears in a coding prompt.
    """

    def __init__(
        self,
        db: Database,
        *,
        registry: Optional[DomainRegistry] = None,
        enable_tools: bool = False,
        max_rules: int = 12,
        max_examples: int = 2,
        tool_timeout: float = 5.0,
    ) -> None:
        self.db = db
        self.registry = registry or DomainRegistry()
        self.examples = ExampleStore(db)
        self.enable_tools = enable_tools
        self.max_rules = max_rules
        self.max_examples = max_examples
        self.tool_timeout = tool_timeout

    def version(self) -> int:
        return self.db.get_policy_version()

    def active_rules(
        self,
        route: Optional[str] = None,
        *,
        exclude_patch_ids: Collection[int] = (),
    ) -> list[str]:
        rows = self.db.list_policy_rules(active_only=True, route=route)
        rules = [
            row["rule_text"]
            for row in rows
            if row["source_patch_id"] not in exclude_patch_ids
        ]
        # Newest rules win when the prompt budget is exceeded.
        return rules[-self.max_rules :]

    def active_tools(
        self,
        route: str,
        *,
        exclude_patch_ids: Collection[int] = (),
    ) -> list[dict[str, Any]]:
        if not self.enable_tools:
            return []
        return [
            tool
            for tool in self.db.list_tools(route=route)
            if tool["source_patch_id"] not in exclude_patch_ids
        ]

    def render_system_prompt(
        self,
        *,
        route: str,
        prompt: Optional[str] = None,
        extra_rules: Optional[list[str]] = None,
        extra_examples: Optional[list[dict[str, str]]] = None,
        extra_tools: Optional[list[dict[str, str]]] = None,
        exclude_patch_ids: Collection[int] = (),
    ) -> str:
        """Build the system prompt for one request.

        ``prompt`` is only needed for tools, which compute their note from
        it. The ``extra_*`` arguments let a candidate patch be tried without
        installing it; ``exclude_patch_ids`` does the opposite and hides an
        installed patch, which is how the audit measures what each one adds.
        """
        lines = list(_BASE_RULES)
        domain = self.registry.get(route)
        lines.append(f"Task type: {route}." + (f" {domain.description}" if domain.description else ""))

        rules = self.active_rules(route, exclude_patch_ids=exclude_patch_ids)
        if extra_rules:
            rules = rules + list(extra_rules)
        if rules:
            lines.append("Learned rules:")
            for index, rule in enumerate(rules, start=1):
                lines.append(f"{index}. {rule}")

        rendered_examples = self.examples.render(
            route=route,
            limit=self.max_examples,
            extra=extra_examples,
            exclude_patch_ids=exclude_patch_ids,
        )
        if rendered_examples:
            lines.append("Reference examples:")
            lines.append(rendered_examples)

        if prompt is not None:
            tools = [
                {"name": tool["name"], "source": tool["source"]}
                for tool in self.active_tools(route, exclude_patch_ids=exclude_patch_ids)
            ]
            if extra_tools:
                tools = tools + list(extra_tools)
            if tools:
                notes = render_tool_notes(run_tools(tools, prompt, timeout=self.tool_timeout))
                if notes:
                    lines.append(notes)

        return "\n".join(lines)

    def apply_prompt_rule(
        self,
        rule_text: str,
        *,
        source_patch_id: Optional[int] = None,
        route: Optional[str] = None,
    ) -> int:
        return self.db.insert_policy_rule(
            rule_text=rule_text,
            source_patch_id=source_patch_id,
            active=True,
            route=route,
        )

    def install_tool(
        self,
        *,
        name: str,
        route: str,
        source: str,
        source_patch_id: Optional[int] = None,
    ) -> int:
        return self.db.insert_tool(
            name=name,
            route=route,
            source=source,
            source_patch_id=source_patch_id,
        )
