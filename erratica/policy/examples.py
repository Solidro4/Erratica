from __future__ import annotations

from typing import Collection, Optional

from erratica.storage.db import Database


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
        source_patch_id: Optional[int] = None,
    ) -> int:
        return self.db.insert_few_shot_example(
            route=route,
            prompt=prompt,
            response=response,
            active=active,
            source_patch_id=source_patch_id,
        )

    def active_examples(
        self,
        route: str,
        limit: int = 2,
        *,
        exclude_patch_ids: Collection[int] = (),
    ) -> list[dict[str, str]]:
        rows = self.db.list_few_shot_examples(route=route, limit=limit + len(exclude_patch_ids))
        kept = [row for row in rows if row["source_patch_id"] not in exclude_patch_ids]
        return kept[:limit]

    def render(
        self,
        route: str,
        limit: int = 2,
        extra: Optional[list[dict[str, str]]] = None,
        *,
        exclude_patch_ids: Collection[int] = (),
    ) -> str:
        examples = self.active_examples(route=route, limit=limit, exclude_patch_ids=exclude_patch_ids)
        seen = {example["prompt"] for example in examples}
        for example in extra or []:
            if example["prompt"] not in seen:
                seen.add(example["prompt"])
                examples.append(example)
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
