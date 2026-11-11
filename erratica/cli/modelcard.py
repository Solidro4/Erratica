from __future__ import annotations

import argparse
from typing import Optional

from erratica import Erratica
from erratica.cli.common import add_db_argument


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="erratica modelcard",
        description="Generate a model card from the Erratica audit trail.",
    )
    add_db_argument(parser)
    parser.add_argument("--name", default="local model", help="Name of the model being published.")
    parser.add_argument("--out", default=None, help="Write the card to this Markdown file.")
    args = parser.parse_args(argv)

    with Erratica(db_path=args.db) as ai:
        if args.out:
            path = ai.write_model_card(args.out, model_name=args.name)
            print(f"Model card written to {path}")
        else:
            print(ai.model_card(model_name=args.name))


if __name__ == "__main__":
    main()
