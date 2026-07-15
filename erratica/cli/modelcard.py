from __future__ import annotations

import argparse

from erratica import Erratica


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a verifiable model card from the Erratica learning database."
    )
    parser.add_argument("--db", default="data/erratica.db", help="Path to the SQLite database.")
    parser.add_argument("--name", default="local model", help="Name of the model being published.")
    parser.add_argument("--out", default=None, help="Optional path to write the card as Markdown.")
    args = parser.parse_args()

    ai = Erratica(db_path=args.db)
    try:
        card = ai.model_card(model_name=args.name)
        if args.out:
            path = ai.write_model_card(args.out, model_name=args.name)
            print(f"Model card written to {path}")
        else:
            print(card)
    finally:
        ai.close()


if __name__ == "__main__":
    main()
