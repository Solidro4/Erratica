from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from erratica.textutil import normalize_prompt
from erratica.policy.prompt_rules import PromptPolicy
from erratica.storage.db import Database


@dataclass
class ExportReport:
    path: Path
    examples_written: int
    corrections: int
    few_shot_examples: int
    held_out: int = 0


class TrainingDataExporter:
    """Turns validated experience into a fine-tuning dataset.

    Every human correction and every promoted few-shot example becomes a
    chat-format JSONL record (system / user / assistant), ready for LoRA or
    QLoRA fine-tuning of a small local model.

    Corrections written by a critic model are left out unless
    ``include_self_generated`` is set: training a model on another model's
    unverified guesses is how errors get baked into weights.
    """

    def __init__(self, db: Database, policy: PromptPolicy) -> None:
        self.db = db
        self.policy = policy

    def export(
        self,
        path: str | Path,
        *,
        exclude_prompts: Iterable[str] = (),
        include_self_generated: bool = False,
    ) -> ExportReport:
        """Write the dataset. ``exclude_prompts`` keeps held-out prompts out
        of the training data so they can be used to judge the fine-tune."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        excluded = {normalize_prompt(prompt) for prompt in exclude_prompts}
        sources = None if include_self_generated else ("human",)

        records: list[dict] = []
        seen: set[tuple[str, str]] = set()
        corrections = 0
        few_shots = 0
        held_out = 0

        for row in self.db.list_corrections(sources=sources, limit=100000):
            key = (row["prompt"], row["reference_response"])
            if key in seen:
                continue
            seen.add(key)
            if normalize_prompt(row["prompt"]) in excluded:
                held_out += 1
                continue
            records.append(
                self._to_record(
                    route=row["route"],
                    prompt=row["prompt"],
                    response=row["reference_response"],
                )
            )
            corrections += 1

        for example in self.db.list_all_few_shot_examples():
            key = (example["prompt"], example["response"])
            if key in seen:
                continue
            seen.add(key)
            if normalize_prompt(example["prompt"]) in excluded:
                held_out += 1
                continue
            records.append(
                self._to_record(
                    route=example["route"],
                    prompt=example["prompt"],
                    response=example["response"],
                )
            )
            few_shots += 1

        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        return ExportReport(
            path=output_path,
            examples_written=len(records),
            corrections=corrections,
            few_shot_examples=few_shots,
            held_out=held_out,
        )

    def _to_record(self, *, route: str, prompt: str, response: str) -> dict:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": self.policy.render_system_prompt(route=route, prompt=prompt),
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        }
