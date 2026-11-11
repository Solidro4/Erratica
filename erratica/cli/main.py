from __future__ import annotations

import argparse
import sys
from typing import Optional

from erratica import __version__
from erratica.cli import benchmark, dashboard, demo, export, modelcard
from erratica.cli.common import add_db_argument, add_model_arguments, describe_decision, open_engine

_USAGE = """usage: erratica <command> [options]

Commands:
  demo         Walk through the learning loop on a small support dataset
  benchmark    Measure what the loop changes on a replay dataset
  cycle        Run one learning cycle on a database
  audit        Measure each live patch and roll back the ones that add nothing
  consolidate  Learning cycle + audit in one pass (for a nightly scheduled task)
  rollback     Undo one promoted patch by id
  dashboard    Show what a database has learned
  modelcard    Generate a model card from the audit trail
  export       Export corrections as a fine-tuning dataset

Run `erratica <command> --help` for the options of one command.
"""


def _cycle(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="erratica cycle", description="Run one learning cycle.")
    add_db_argument(parser)
    add_model_arguments(parser)
    parser.add_argument("--force", action="store_true", help="Re-examine clusters with no new feedback.")
    args = parser.parse_args(argv)
    with open_engine(args) as ai:
        decisions = ai.run_learning_cycle(force=args.force)
    if not decisions:
        print("No patch was proposed (no new learnable failures).")
    for decision in decisions:
        print(describe_decision(decision))


def _print_audit(findings: list) -> None:
    if not findings:
        print("No live patches to audit.")
    for finding in findings:
        print(
            f"patch #{finding.patch_id} {finding.patch_type} [{finding.route or 'all routes'}]: "
            f"contributes {finding.contribution:+.3f} over {finding.cases} cases -> {finding.action}"
        )


def _audit(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="erratica audit",
        description="Switch each live patch off, measure the difference, and prune dead weight.",
    )
    add_db_argument(parser)
    add_model_arguments(parser)
    parser.add_argument("--no-prune", action="store_true", help="Report only; roll nothing back.")
    args = parser.parse_args(argv)
    with open_engine(args) as ai:
        _print_audit(ai.run_audit(prune=not args.no_prune))


def _consolidate(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="erratica consolidate",
        description="One offline pass: learning cycle, then audit.",
    )
    add_db_argument(parser)
    add_model_arguments(parser)
    args = parser.parse_args(argv)
    with open_engine(args) as ai:
        report = ai.consolidate()
    print("Learning cycle:")
    if not report.decisions:
        print("No patch was proposed (no new learnable failures).")
    for decision in report.decisions:
        print(describe_decision(decision))
    print()
    print("Audit:")
    _print_audit(report.audit)


def _rollback(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="erratica rollback", description="Undo one promoted patch.")
    add_db_argument(parser)
    parser.add_argument("patch_id", type=int)
    parser.add_argument("--reason", default="manual rollback")
    args = parser.parse_args(argv)
    with open_engine(args) as ai:
        done = ai.rollback(args.patch_id, reason=args.reason)
    print(f"Patch #{args.patch_id} rolled back." if done else f"Patch #{args.patch_id} is not live.")
    if not done:
        sys.exit(1)


_COMMANDS = {
    "demo": demo.main,
    "benchmark": benchmark.main,
    "cycle": _cycle,
    "audit": _audit,
    "consolidate": _consolidate,
    "rollback": _rollback,
    "dashboard": dashboard.main,
    "modelcard": modelcard.main,
    "export": export.main,
}


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_USAGE)
        return
    if argv[0] in ("-V", "--version"):
        print(f"erratica {__version__}")
        return
    command = _COMMANDS.get(argv[0])
    if command is None:
        print(f"erratica: unknown command '{argv[0]}'\n", file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        sys.exit(2)
    command(argv[1:])


if __name__ == "__main__":
    main()
