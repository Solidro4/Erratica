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
from erratica.learning.clustering import is_failing

_BAD_RULE = "Respond in a single short sentence, avoid detail, and avoid timelines."


def render_benchmark(
    dataset_path: Optional[str | Path] = None,
    args: Optional[argparse.Namespace] = None,
) -> str:
    """Measure what the learning loop changes for a model on a dataset.

    The dataset's training prompts are asked one by one. Good answers are
    approved and bad ones get the dataset's feedback, as a user would give
    it. Then one learning cycle runs.

    Two things are reported. The headline number is the score on the
    dataset's test prompts, which the system never received feedback on,
    before and after learning. The second table replays the recorded
    failures with the learned patches switched off and on.

    Works with any model: pass ``--ollama`` or ``--gguf`` to measure a real
    one.
    """
    args = args or argparse.Namespace()
    train = load_cases(dataset_path, split="train")
    test = load_cases(dataset_path, split="test")
    with tempfile.TemporaryDirectory() as tempdir:
        ai = open_engine(args, db_path=Path(tempdir) / "benchmark.db")
        try:
            lines = ["=== Erratica Benchmark ==="]
            if uses_simulator(args):
                lines.append(SIMULATOR_NOTE)

            test_cases = [to_replay_case(ai, case, -(index + 1)) for index, case in enumerate(test)]
            test_before = ai.replay_runner.run(test_cases)

            recorded = record_cases(ai, train)
            failures = sum(1 for item in recorded if item.failed)
            decisions = ai.run_learning_cycle()
            route = recorded[0].result.route
            control = ai.propose_rule(_BAD_RULE, route=route)

            lines += [
                "",
                f"Training prompts: {len(train)} asked, {failures} failed and got feedback, "
                f"{len(train) - failures} were good and approved",
                "",
                "Learning cycle decisions:",
            ]
            for decision in decisions:
                lines.append("- " + describe_decision(decision))
            if not decisions:
                lines.append("- none (nothing learnable, or nothing worth proposing)")
            lines += ["", "Control (a deliberately bad rule):", "- " + describe_decision(control)]

            if test_cases:
                test_after = ai.replay_runner.run(test_cases)
                unseen = ai.evaluator.evaluate(test_cases, test_before, test_after)
                lines += ["", "Unseen test prompts (no feedback was ever given on these):"]
                lines += _case_lines(unseen, {case.interaction_id: case.prompt for case in test_cases})
                lines.append(_summary_line(unseen))

            promoted = {decision.patch_id for decision in decisions if decision.accepted}
            replay_cases = ai.dataset_builder.build(
                [row for row in ai.memory.failed_interactions() if is_failing(row)]
            )
            replay_cases.sort(key=lambda case: case.interaction_id)
            if replay_cases:
                without = ai.replay_runner.run(replay_cases, exclude_patch_ids=promoted)
                with_patches = ai.replay_runner.run(replay_cases)
                seen = ai.evaluator.evaluate(replay_cases, without, with_patches)
                lines += ["", "Recorded failures (learned patches off -> on):"]
                lines += _case_lines(seen, {case.interaction_id: case.prompt for case in replay_cases})
                lines.append(_summary_line(seen))
                lines.append(
                    "(A case that is already high with patches off was fixed by experience recall.)"
                )
            return "\n".join(lines)
        finally:
            ai.close()


def _case_lines(summary, prompts: dict) -> list[str]:
    return [
        f"- {delta['before']:.2f} -> {delta['after']:.2f}  {_shorten(prompts[delta['interaction_id']])}"
        for delta in summary.case_deltas
    ]


def _summary_line(summary) -> str:
    return (
        f"Average: {summary.score_before:.2f} -> {summary.score_after:.2f} ({summary.delta:+.2f}) "
        f"over {summary.scored} cases; {summary.improved} improved, {summary.worsened} worse"
    )


def _shorten(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="erratica benchmark",
        description="Measure what the learning loop changes on a replay dataset.",
    )
    parser.add_argument(
        "--dataset",
        default="support",
        help="'support', 'arithmetic', or the path to a JSON dataset (default: support).",
    )
    add_model_arguments(parser)
    args = parser.parse_args(argv)
    print(render_benchmark(args.dataset, args))


if __name__ == "__main__":
    main()
