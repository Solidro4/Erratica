from __future__ import annotations

import argparse
from typing import Optional

from erratica import Erratica
from erratica.cli.common import add_db_argument


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="erratica export",
        description="Export recorded corrections as a chat-format fine-tuning dataset.",
    )
    add_db_argument(parser)
    parser.add_argument("--out", default="data/erratica_sft.jsonl", help="Output JSONL path.")
    parser.add_argument(
        "--include-self-generated",
        action="store_true",
        help="Also export corrections written by a critic model (unverified).",
    )
    args = parser.parse_args(argv)

    with Erratica(db_path=args.db) as ai:
        report = ai.export_training_data(
            args.out,
            include_self_generated=args.include_self_generated,
        )

    print("=== Erratica Training Export ===")
    print(f"Output: {report.path}")
    print(f"Examples written: {report.examples_written}")
    print(f"- from corrections: {report.corrections}")
    print(f"- from promoted few-shot examples: {report.few_shot_examples}")
    print()
    print("This file is every correction. To fine-tune and verify in one step, with")
    print("some failures held out of training to judge the result, use")
    print("Erratica.run_weight_cycle(trainer) instead.")


if __name__ == "__main__":
    main()
