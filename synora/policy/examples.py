from __future__ import annotations

from synora.storage.db import Database


class ExampleStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add_example(
        self,
        *,
        route: str,
        prompt: str,
        response: str,
        active: bool = True,
    ) -> int:
        return self.db.insert_few_shot_example(
            route=route,
            prompt=prompt,
            response=response,
            active=active,
        )

    def active_examples(self, route: str, limit: int = 2) -> list[dict[str, str]]:
        return self.db.list_few_shot_examples(route=route, limit=limit)

    def render(self, route: str, limit: int = 2) -> str:
        examples = self.active_examples(route=route, limit=limit)
        if not examples:
            return ""
        blocks: list[str] = []
        for example in examples:
            blocks.append(
                "\n".join(
                    [
                        "Example input:",
                        example["prompt"],
                        "Example output:",
                        example["response"],
                    ]
                )
            )
        return "\n\n".join(blocks)
