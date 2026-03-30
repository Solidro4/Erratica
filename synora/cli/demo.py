from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from synora import Synora

_DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "datasets" / "support_replay_cases.json"


def load_cases(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_demo(dataset_path: str | Path) -> str:
    cases = load_cases(dataset_path)
    with tempfile.TemporaryDirectory() as tempdir:
        ai = Synora(db_path=Path(tempdir) / "support_demo.db")
        try:
            focus_case = cases[0]
            first = ai.generate(focus_case["prompt"])
            ai.learn_from_feedback(
                interaction_id=first.interaction_id,
                rating=-1,
                issue_type=focus_case["issue_type"],
                required_terms=focus_case.get("required_terms"),
                preferred_format=focus_case.get("preferred_format"),
                notes=focus_case["feedback"],
            )

            for case in cases[1:]:
                result = ai.generate(case["prompt"])
                ai.learn_from_feedback(
                    interaction_id=result.interaction_id,
                    rating=-1,
                    issue_type=case["issue_type"],
                    required_terms=case.get("required_terms"),
                    preferred_format=case.get("preferred_format"),
                    notes=case["feedback"],
                )

            decisions = ai.run_learning_cycle(limit=len(cases))
            decision = decisions[0]
            improved = ai.generate(focus_case["prompt"])

            lines = [
                "=== Synora Demo ===",
                "",
                "Prompt:",
                focus_case["prompt"],
                "",
                "Before:",
                first.text,
                "",
                "Feedback:",
                focus_case["feedback"],
                "",
                "Applying patch...",
                "",
                f"Replaying {len(cases)} failures...",
                f"Score before: {decision.score_before:.2f}",
                f"Score after:  {decision.score_after:.2f}",
                "",
                f"Patch: {decision.rule_text}",
                "",
                "Status: " + ("PROMOTED" if decision.accepted else "REJECTED"),
                "",
                "After:",
                improved.text,
            ]
            return "\n".join(lines)
        finally:
            ai.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the support-ticket Synora demo.")
    parser.add_argument("--dataset", default=str(_DEFAULT_DATASET), help="Path to a replay dataset JSON file.")
    args = parser.parse_args()
    print(render_demo(args.dataset))


if __name__ == "__main__":
    main()
