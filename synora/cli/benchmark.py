from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from synora import Synora
from synora.cli.demo import _DEFAULT_DATASET, load_cases
from synora.learning.patcher import PatchProposal


def render_benchmark(dataset_path: str | Path) -> str:
    cases = load_cases(dataset_path)
    with tempfile.TemporaryDirectory() as tempdir:
        ai = Synora(db_path=Path(tempdir) / "benchmark.db")
        try:
            for case in cases:
                result = ai.generate(case["prompt"])
                ai.learn_from_feedback(
                    interaction_id=result.interaction_id,
                    rating=-1,
                    issue_type=case["issue_type"],
                    required_terms=case.get("required_terms"),
                    preferred_format=case.get("preferred_format"),
                    ideal_response=case.get("ideal_response"),
                    notes=case["feedback"],
                )

            failures = ai.memory.failed_interactions(limit=len(cases))
            clusters = ai.clusterer.cluster(failures)
            if not clusters:
                return "No replay cases available."

            cluster = clusters[0]
            replay_cases = ai.dataset_builder.build_from_cluster(cluster)
            baseline_results = ai.replay_runner.run(replay_cases)

            accepted_proposal = ai.patcher.propose(cluster)
            accepted_results = ai.replay_runner.run(replay_cases, extra_rules=[accepted_proposal.rule_text])
            accepted_evaluation = ai.evaluator.evaluate(replay_cases, baseline_results, accepted_results)

            rejected_proposal = PatchProposal(
                patch_type="prompt_rule",
                target="system_prompt",
                rule_text="Respond in a single short sentence, avoid detail, and avoid timelines.",
                rationale="Control behavior update used to verify that Synora rejects degraded policies.",
                source_issue_type="control",
            )
            rejected_results = ai.replay_runner.run(replay_cases, extra_rules=[rejected_proposal.rule_text])
            rejected_evaluation = ai.evaluator.evaluate(replay_cases, baseline_results, rejected_results)

            rejected_decision = ai.promoter.promote(rejected_proposal, rejected_evaluation)
            accepted_decision = ai.promoter.promote(accepted_proposal, accepted_evaluation)

            lines = [
                "=== Synora Benchmark ===",
                "",
                "Learning flow:",
                "User Prompt",
                "  |",
                "  v",
                "Initial Response",
                "  |",
                "  v",
                "Feedback",
                "  |",
                "  v",
                "Behavior Update Proposal",
                "  |",
                "  v",
                "Replay on Past Failures",
                "  |",
                "  v",
                "Score Comparison",
                "  |",
                "  v",
                "Promote or Reject",
                "",
                "Cluster summary:",
                f"- {cluster.issue_type}: {cluster.size} failures grouped into one behavior update",
                "",
                "Case improvements:",
            ]

            for case, delta in zip(replay_cases, accepted_evaluation.case_deltas):
                lines.append(
                    f"- {short_case_label(case.prompt)}: {delta['before']:.2f} -> {delta['after']:.2f}"
                )

            lines.extend(
                [
                    "",
                    "Average score:",
                    f"- baseline: {accepted_decision.score_before:.2f}",
                    f"- after accepted behavior update: {accepted_decision.score_after:.2f}",
                    f"- absolute gain: +{accepted_decision.delta:.2f}",
                    f"- relative improvement: {format_relative_gain(accepted_decision.score_before, accepted_decision.score_after)}",
                    "",
                    "Behavior update outcomes:",
                    f"- {rejected_proposal.patch_type} | REJECTED | {rejected_decision.score_before:.2f} -> {rejected_decision.score_after:.2f}",
                    f"  rule: {rejected_proposal.rule_text}",
                    f"- {accepted_proposal.patch_type} | PROMOTED | {accepted_decision.score_before:.2f} -> {accepted_decision.score_after:.2f}",
                    f"  rule: {accepted_proposal.rule_text}",
                ]
            )
            return "\n".join(lines)
        finally:
            ai.close()


def short_case_label(prompt: str) -> str:
    lowered = prompt.lower()
    if "charged twice" in lowered or "duplicate charge" in lowered:
        return "duplicate charge"
    if "billed" in lowered or "subscription" in lowered:
        return "billing issue"
    if "refund" in lowered:
        return "refund delay"
    if "damaged" in lowered or "replacement" in lowered:
        return "damaged item"
    if "tracking" in lowered or "shipment" in lowered:
        return "shipment delay"
    return "support case"


def format_relative_gain(before: float, after: float) -> str:
    denominator = max(before, 0.01)
    gain = ((after - before) / denominator) * 100.0
    return f"+{gain:.0f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Synora benchmark mode.")
    parser.add_argument("--dataset", default=str(_DEFAULT_DATASET), help="Path to a replay dataset JSON file.")
    args = parser.parse_args()
    print(render_benchmark(args.dataset))


if __name__ == "__main__":
    main()
