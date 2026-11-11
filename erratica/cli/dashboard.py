from __future__ import annotations

import argparse
from typing import Optional

from erratica import Erratica
from erratica.cli.common import add_db_argument


def describe_patch(patch: dict) -> str:
    content = patch["content"]
    return (
        content.get("rule_text")
        or content.get("description")
        or f"Reference example for route '{content.get('route', 'general')}'"
    )


def render_dashboard(snapshot: dict) -> str:
    metrics = snapshot["metrics"]
    lines = [
        "Erratica Dashboard",
        f"Policy version: {metrics.get('policy_version', 0)}",
        f"Interactions: {metrics['interactions']}",
        f"Feedback items: {metrics['feedback_items']} "
        f"({metrics.get('approvals', 0)} approvals, {metrics.get('self_review_items', 0)} from self-review)",
        f"Patches: {metrics['promoted_patches']} promoted, {metrics.get('rejected_patches', 0)} rejected, "
        f"{metrics.get('rolled_back_patches', 0)} rolled back",
        f"Live: {metrics['active_rules']} rules, {metrics.get('active_examples', 0)} examples, "
        f"{metrics.get('active_tools', 0)} tools",
        f"Quarantined clusters: {metrics.get('quarantined_clusters', 0)}",
        "",
        "Issue clusters:",
    ]

    if snapshot["issue_clusters"]:
        for cluster in snapshot["issue_clusters"]:
            lines.append(f"- {cluster['issue_type']}: {cluster['count']}")
    else:
        lines.append("- none yet")

    lines += ["", "Live rules:"]
    if snapshot["active_rules"]:
        for rule in snapshot["active_rules"]:
            lines.append(f"- #{rule['id']} [{rule.get('route') or 'all routes'}] {rule['rule_text']}")
    else:
        lines.append("- none yet")

    if snapshot.get("active_tools"):
        lines += ["", "Live tools:"]
        for tool in snapshot["active_tools"]:
            lines.append(f"- {tool['name']} [{tool['route']}] (patch #{tool['source_patch_id']})")

    lines += ["", "Recent patches:"]
    if snapshot["recent_patches"]:
        for patch in snapshot["recent_patches"]:
            delta = ""
            if patch["score_before"] is not None and patch["score_after"] is not None:
                delta = f" ({patch['score_after'] - patch['score_before']:+.2f})"
            validation = patch.get("details", {}).get("validation")
            suffix = f" [{validation}]" if validation else ""
            lines.append(
                f"- #{patch['id']} {patch['patch_type']} {patch['status']}{delta}{suffix}: "
                f"{describe_patch(patch)}"
            )
    else:
        lines.append("- none yet")

    lines += ["", "Mistake triage:"]
    triage_entries = snapshot.get("recent_triage", [])
    if triage_entries:
        for entry in triage_entries:
            lines.append(
                f"- {entry.get('route') or '?'}/{entry['issue_type']} | {entry['verdict'].upper()} | "
                f"score {entry['score']:.2f} | {entry['case_count']} case(s)"
            )
    else:
        lines.append("- none yet")

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="erratica dashboard",
        description="Show what an Erratica database has learned.",
    )
    add_db_argument(parser)
    args = parser.parse_args(argv)

    with Erratica(db_path=args.db) as ai:
        print(render_dashboard(ai.dashboard()))


if __name__ == "__main__":
    main()
