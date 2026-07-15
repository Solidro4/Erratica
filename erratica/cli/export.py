from __future__ import annotations

import argparse

from erratica import Erratica


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export validated Erratica experience as a fine-tuning dataset."
    )
    parser.add_argument("--db", default="data/erratica.db", help="Path to the SQLite database.")
    parser.add_argument(
        "--out",
        default="data/erratica_sft.jsonl",
        help="Output JSONL path (chat format, ready for LoRA/QLoRA fine-tuning).",
    )
    args = parser.parse_args()

    ai = Erratica(db_path=args.db)
    try:
        report = ai.export_training_data(args.out)
    finally:
        ai.close()

    print("=== Erratica Training Export ===")
    print(f"Output: {report.path}")
    print(f"Examples written: {report.examples_written}")
    print(f"- from user corrections: {report.corrections}")
    print(f"- from promoted few-shot examples: {report.few_shot_examples}")
    print()
    print("Fine-tune a small local model on this file with unsloth, axolotl,")
    print("or llama.cpp LoRA training, then load the tuned .gguf back into Erratica.")


if __name__ == "__main__":
    main()
