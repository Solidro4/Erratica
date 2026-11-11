from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Optional

from erratica.cli.common import (
    SIMULATOR_NOTE,
    add_model_arguments,
    describe_decision,
    load_cases,
    open_engine,
    record_cases,
    to_replay_case,
    uses_simulator,
)

_SIMILAR_PROMPT = "Customer says order #88221 was returned last week and the refund still has not arrived."
_NEW_PROMPT = "Customer says the package for order #70555 never showed up and asks what happens now."
_BAD_RULE = "Respond in a single short sentence, avoid detail, and avoid timelines."


def render_demo(
    dataset_path: Optional[str | Path] = None,
    args: Optional[argparse.Namespace] = None,
) -> str:
    args = args or argparse.Namespace()
    cases = load_cases(dataset_path)
    with tempfile.TemporaryDirectory() as tempdir:
        ai = open_engine(args, db_path=Path(tempdir) / "demo.db")
        try:
            focus = cases[0]
            lines = ["=== Erratica Demo ==="]
            if uses_simulator(args):
                lines.append(SIMULATOR_NOTE)

            # 1. One failure and its correction.
            first = record_cases(ai, [focus])[0].result
            lines += [
                "",
                "1. A failure",
                f"Prompt:   {focus['prompt']}",
                f"Answer:   {first.text}",
                f"Feedback: {focus['feedback']}",
            ]

            # 2. Instant: the correction is recalled for a similar prompt.
            similar = ai.generate(_SIMILAR_PROMPT)
            similar_score = ai.evaluator.score_case(
                to_replay_case(ai, dict(focus, prompt=_SIMILAR_PROMPT), similar.interaction_id),
                similar.text,
            )
            lines += [
                "",
                "2. Instant recall (no learning cycle yet)",
                f"Prompt:   {_SIMILAR_PROMPT}",
                "Answer:",
                similar.text,
            ]
            if similar_score is not None and similar_score >= 0.8:
                ai.approve(similar.interaction_id)
                lines.append("(approved by the user: later patches must not break this answer)")
            else:
                lines.append("(recall alone was not enough for this one)")

            # 3. More failures arrive, then a learning cycle.
            later = record_cases(ai, cases[1:])
            failures = 1 + sum(1 for item in later if item.failed)
            decisions = ai.run_learning_cycle()
            triage = ai.dashboard()["recent_triage"]
            lines += [
                "",
                f"3. Learning cycle ({len(cases)} prompts asked so far, {failures} failed and got feedback)",
            ]
            for entry in triage:
                lines.append(
                    f"Triage: {entry['route']}/{entry['issue_type']} -> {entry['verdict'].upper()} "
                    f"(score {entry['score']:.2f}, {entry['case_count']} cases)"
                )
            for decision in decisions:
                lines.append(describe_decision(decision))
            if not decisions:
                lines.append("No patch was proposed.")

            # 4. A prompt the system has never seen.
            fresh = ai.generate(_NEW_PROMPT)
            lines += [
                "",
                "4. A new prompt after learning",
                f"Prompt:   {_NEW_PROMPT}",
                "Answer:",
                fresh.text,
            ]

            # 5. The gate also says no.
            control = ai.propose_rule(_BAD_RULE, route="support")
            lines += ["", "5. A bad rule submitted by hand", describe_decision(control)]

            # 6. Audit: what is each live patch worth?
            lines += ["", "6. Audit (each live patch switched off and measured)"]
            for finding in ai.run_audit():
                lines.append(
                    f"patch #{finding.patch_id} {finding.patch_type}: contributes "
                    f"{finding.contribution:+.2f} over {finding.cases} cases -> {finding.action}"
                )
            return "\n".join(lines)
        finally:
            ai.close()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="erratica demo",
        description="Walk through the Erratica learning loop on support tickets.",
    )
    parser.add_argument("--dataset", default="support", help="Bundled dataset name or JSON path.")
    add_model_arguments(parser)
    args = parser.parse_args(argv)
    print(render_demo(args.dataset, args))


if __name__ == "__main__":
    main()
