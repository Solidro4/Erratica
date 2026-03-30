from __future__ import annotations

import argparse

from synora import Synora


def render_dashboard(snapshot: dict) -> str:
    metrics = snapshot["metrics"]
    lines = [
        "Synora Dashboard",
        f"Interactions: {metrics['interactions']}",
        f"Feedback items: {metrics['feedback_items']}",
        f"Promoted patches: {metrics['promoted_patches']}",
        f"Active rules: {metrics['active_rules']}",
        "",
        "Issue clusters:",
    ]

    if snapshot["issue_clusters"]:
        for cluster in snapshot["issue_clusters"]:
            lines.append(f"- {cluster['issue_type']}: {cluster['count']}")
    else:
        lines.append("- none yet")

    lines.append("")
    lines.append("Active prompt rules:")
    if snapshot["active_rules"]:
        for rule in snapshot["active_rules"]:
            lines.append(f"- #{rule['id']}: {rule['rule_text']}")
    else:
        lines.append("- none yet")

    lines.append("")
    lines.append("Recent patches:")
    if snapshot["recent_patches"]:
        for patch in snapshot["recent_patches"]:
            content = patch["content"].get("rule_text", "")
            delta = 0.0
            if patch["score_before"] is not None and patch["score_after"] is not None:
                delta = patch["score_after"] - patch["score_before"]
            lines.append(f"- #{patch['id']} {patch['status']} ({delta:+.2f}): {content}")
    else:
        lines.append("- none yet")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the Synora learning dashboard.")
    parser.add_argument("--db", default="data/synora.db", help="Path to the SQLite database.")
    args = parser.parse_args()

    ai = Synora(db_path=args.db)
    try:
        print(render_dashboard(ai.dashboard()))
    finally:
        ai.close()


if __name__ == "__main__":
    main()
