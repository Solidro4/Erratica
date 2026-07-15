from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from erratica.policy.prompt_rules import PromptPolicy
from erratica.storage.db import Database


@dataclass
class ExportReport:
    path: Path
    examples_written: int
    corrections: int
    few_shot_examples: int


class TrainingDataExporter:
    """Turns validated experience into a fine-tuning dataset.

    Every user correction and every promoted few-shot example becomes a
    chat-format JSONL record (system / user / assistant), ready for LoRA or
    QLoRA fine-tuning of small local models with tools like llama.cpp,
    unsloth, or axolotl. This is how prompt-level learning graduates into
    weight-level learning without a datacenter.
    """

    def __init__(self, db: Database, policy: PromptPolicy) -> None:
        self.db = db
        self.policy = policy

    def export(self, path: str | Path) -> ExportReport:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records: list[dict] = []
        seen: set[tuple[str, str]] = set()
        corrections = 0
        few_shots = 0

        for row in self.db.list_corrections():
            record = self._to_record(
                route=row["route"],
                prompt=row["prompt"],
                response=row["reference_response"],
            )
            key = (row["prompt"], row["reference_response"])
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
            corrections += 1

        for example in self.db.list_all_few_shot_examples():
            key = (example["prompt"], example["response"])
            if key in seen:
                continue
            seen.add(key)
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
        )

    def _to_record(self, *, route: str, prompt: str, response: str) -> dict:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": self.policy.render_system_prompt(route=route),
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        }
