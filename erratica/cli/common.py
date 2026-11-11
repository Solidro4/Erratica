from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from erratica import Erratica
from erratica.learning.replay import ReplayCase

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "support_replay_cases.json"
SIMULATOR_NOTE = (
    "Model: built-in deterministic simulator. These numbers show the loop working, "
    "not how much a real model improves."
)


DATASETS = {
    "support": DEFAULT_DATASET,
    "arithmetic": DEFAULT_DATASET.with_name("arithmetic_cases.json"),
}


def load_cases(path: Optional[str | Path] = None, *, split: Optional[str] = "train") -> list[dict]:
    """Load a replay dataset: a bundled one by name, or a JSON file by path.

    Entries marked ``"split": "test"`` are prompts the system is never given
    feedback on; they are only used to measure generalisation. ``split``
    selects "train", "test", or (with None) everything.
    """
    source = DATASETS.get(str(path)) if path else DEFAULT_DATASET
    cases = json.loads(Path(source or path).read_text(encoding="utf-8"))
    if split is None:
        return cases
    return [case for case in cases if case.get("split", "train") == split]


def add_db_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default="data/erratica.db", help="Path to the SQLite database.")


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model (default: the built-in simulator)")
    group.add_argument("--gguf", help="Path to a local .gguf model (needs erratica[llama]).")
    group.add_argument("--ollama", help="Name of an Ollama model to answer with, e.g. llama3.2:3b.")
    group.add_argument(
        "--proposer",
        help="Ollama model that writes candidate rules and tools (default: none).",
    )
    group.add_argument(
        "--tools",
        action="store_true",
        help="Allow learned Python tools (runs model-written code in a restricted subprocess).",
    )
    group.add_argument("--ollama-host", default="http://localhost:11434")
    group.add_argument(
        "--no-think",
        action="store_true",
        help="Ask Ollama reasoning models to answer without their thinking phase.",
    )


def _think(args: argparse.Namespace) -> Optional[bool]:
    return False if getattr(args, "no_think", False) else None


def build_model(args: argparse.Namespace) -> Any:
    if getattr(args, "gguf", None):
        from erratica.engine.llama_cpp_adapter import LlamaCppModel

        return LlamaCppModel(model_path=args.gguf)
    if getattr(args, "ollama", None):
        from erratica.engine.ollama_adapter import OllamaModel

        return OllamaModel(args.ollama, host=args.ollama_host, think=_think(args))
    return None


def open_engine(args: argparse.Namespace, db_path: Optional[str | Path] = None) -> Erratica:
    proposer = None
    if getattr(args, "proposer", None):
        from erratica.engine.ollama_adapter import OllamaModel

        proposer = OllamaModel(
            args.proposer,
            host=args.ollama_host,
            max_tokens=1024,
            think=_think(args),
        )
    return Erratica(
        db_path=db_path or args.db,
        model=build_model(args),
        proposer=proposer,
        enable_tools=bool(getattr(args, "tools", False)),
    )


def uses_simulator(args: argparse.Namespace) -> bool:
    return not (getattr(args, "gguf", None) or getattr(args, "ollama", None))


@dataclass
class Recorded:
    result: Any
    score: Optional[float]
    failed: bool


def to_replay_case(ai: Erratica, case: dict, interaction_id: int, response: str = "") -> ReplayCase:
    return ReplayCase(
        interaction_id=interaction_id,
        prompt=case["prompt"],
        baseline_response=response,
        route=case.get("route") or ai.router.select_route(case["prompt"]),
        issue_type=case["issue_type"],
        required_terms=list(case.get("required_terms") or []),
        preferred_format=case.get("preferred_format"),
        ideal_response=case.get("ideal_response"),
    )


def record_cases(ai: Erratica, cases: list[dict], *, pass_score: float = 0.8) -> list[Recorded]:
    """Ask each dataset prompt and give feedback the way a user would.

    The answer is scored against what the dataset says a good answer needs.
    A good answer is approved; a bad one gets the dataset's feedback. So a
    model that already handles a prompt is never told it failed.
    """
    recorded = []
    for case in cases:
        result = ai.generate(case["prompt"], route=case.get("route"))
        score = ai.evaluator.score_case(
            to_replay_case(ai, case, result.interaction_id, result.text),
            result.text,
        )
        failed = score is None or score < pass_score
        if failed:
            ai.learn_from_feedback(
                interaction_id=result.interaction_id,
                rating=-1,
                issue_type=case["issue_type"],
                required_terms=case.get("required_terms"),
                preferred_format=case.get("preferred_format"),
                ideal_response=case.get("ideal_response"),
                notes=case.get("feedback"),
            )
        else:
            ai.approve(result.interaction_id)
        recorded.append(Recorded(result=result, score=score, failed=failed))
    return recorded


def describe_decision(decision: Any) -> str:
    status = "PROMOTED" if decision.accepted else "REJECTED"
    scope = {
        "holdout": f"{decision.cases_scored} held-out case(s)",
        "in_sample": f"{decision.cases_scored} in-sample case(s)",
        "manual": f"{decision.cases_scored} recorded failure(s)",
    }.get(decision.validation, "not replayed")
    line = (
        f"{decision.patch_type} | {status} | "
        f"{decision.score_before:.2f} -> {decision.score_after:.2f} on {scope}"
    )
    return f"{line}\n  {decision.rule_text}\n  reason: {decision.reason}"
