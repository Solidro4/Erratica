from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

from erratica.learning.clustering import is_failing
from erratica.learning.evaluator import PatchEvaluator
from erratica.learning.exporter import ExportReport, TrainingDataExporter
from erratica.learning.replay import ReplayCase, ReplayDatasetBuilder, ReplayRunner
from erratica.learning.splits import distinct_prompts, split_cases
from erratica.storage.db import Database


class ModelTrainer(Protocol):
    """Anything that turns a JSONL dataset into a new candidate model."""

    def train(self, dataset_path: Path) -> Any:
        ...


class CommandTrainer:
    """Runs any external fine-tuning command and loads what it produced.

    Erratica does not ship a GPU training stack. This adapter lets you plug
    in the one you already use (an unsloth or PEFT script, llama.cpp
    tooling, a remote job) as a command line::

        trainer = CommandTrainer(
            ["python", "train_lora.py", "--data", "{dataset}", "--out", "{output}"],
            output_path="models/tuned.gguf",
            loader=lambda path: LlamaCppModel(model_path=str(path)),
        )

    ``{dataset}`` and ``{output}`` are replaced with the exported dataset
    path and ``output_path``. The command must exit 0 and leave a file at
    ``output_path``, or the cycle fails loudly instead of loading stale
    weights.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        output_path: str | Path,
        loader: Callable[[Path], Any],
        timeout: Optional[float] = None,
        cwd: Optional[str | Path] = None,
    ) -> None:
        self.command = list(command)
        self.output_path = Path(output_path)
        self.loader = loader
        self.timeout = timeout
        self.cwd = cwd

    def train(self, dataset_path: Path) -> Any:
        previous_mtime = self.output_path.stat().st_mtime_ns if self.output_path.exists() else None
        command = [
            part.replace("{dataset}", str(dataset_path)).replace("{output}", str(self.output_path))
            for part in self.command
        ]
        subprocess.run(command, check=True, timeout=self.timeout, cwd=self.cwd)
        if not self.output_path.exists():
            raise RuntimeError(f"training command finished but {self.output_path} does not exist")
        if previous_mtime is not None and self.output_path.stat().st_mtime_ns == previous_mtime:
            raise RuntimeError(f"training command finished but did not update {self.output_path}")
        return self.loader(self.output_path)


@dataclass
class WeightCycleReport:
    accepted: bool
    reason: str
    score_before: float
    score_after: float
    dataset: Optional[ExportReport] = None
    # Held-out failures the candidate was judged on.
    holdout_cases: int = 0
    # Change on answers users approved; negative means the fine-tune forgot.
    wins_delta: Optional[float] = None
    # Scores on the training prompts, for comparison only. A large gap
    # between these and the held-out scores means the fine-tune memorised.
    train_score_before: Optional[float] = None
    train_score_after: Optional[float] = None

    @property
    def delta(self) -> float:
        return self.score_after - self.score_before


class WeightLoop:
    """Closes the loop from validated experience to model weights.

    A fine-tune is the easiest change to fool yourself about: train on the
    corrections, test on the same prompts, and a model that only memorised
    them looks perfect. So the candidate is judged on failures whose
    corrections were kept out of the training data, and it must not get
    worse on answers users approved. If there are too few distinct failures
    to hold any out, the cycle refuses to run.
    """

    def __init__(
        self,
        db: Database,
        exporter: TrainingDataExporter,
        dataset_builder: ReplayDatasetBuilder,
        evaluator: PatchEvaluator,
        replay_runner: ReplayRunner,
        *,
        min_delta: float = 0.05,
        regression_tolerance: float = 0.02,
        max_case_drop: float = 0.3,
        min_holdout: int = 2,
    ) -> None:
        self.db = db
        self.exporter = exporter
        self.dataset_builder = dataset_builder
        self.evaluator = evaluator
        self.replay_runner = replay_runner
        self.min_delta = min_delta
        self.regression_tolerance = regression_tolerance
        self.max_case_drop = max_case_drop
        self.min_holdout = min_holdout

    def run(
        self,
        current_model: Any,
        trainer: ModelTrainer | Callable[[Path], Any],
        *,
        dataset_path: str | Path = Path("data") / "erratica_sft.jsonl",
        replay_limit: int = 500,
        include_self_generated: bool = False,
    ) -> tuple[WeightCycleReport, Any]:
        """Returns the cycle report and the model to use going forward."""
        failures = [
            row for row in self.db.list_failed_interactions(limit=replay_limit) if is_failing(row)
        ]
        cases = self.dataset_builder.build(failures)
        split = split_cases(cases)
        if not split.is_holdout or distinct_prompts(split.holdout) < self.min_holdout:
            return (
                self._refuse(
                    "not enough distinct failures to hold some out "
                    f"(have {distinct_prompts(cases)}, need at least {self.min_holdout + 1})"
                ),
                current_model,
            )

        export = self.exporter.export(
            dataset_path,
            exclude_prompts=[case.prompt for case in split.holdout],
            include_self_generated=include_self_generated,
        )
        if export.examples_written == 0:
            return (
                self._refuse("no reference answers to fine-tune on yet", dataset=export),
                current_model,
            )

        train = trainer.train if hasattr(trainer, "train") else trainer
        candidate_model = train(Path(dataset_path))

        evaluation = self._compare(split.holdout, current_model, candidate_model)
        train_evaluation = self._compare(split.train, current_model, candidate_model)
        wins = self.dataset_builder.build(self.db.list_wins(limit=replay_limit))
        wins_evaluation = self._compare(wins, current_model, candidate_model) if wins else None
        has_wins = wins_evaluation is not None and wins_evaluation.scored > 0

        if evaluation.scored == 0:
            accepted, reason = False, "no held-out case had a checkable signal"
        elif evaluation.delta <= self.min_delta:
            accepted, reason = False, "fine-tuned model did not improve on held-out failures"
        elif has_wins and wins_evaluation.delta < -self.regression_tolerance:
            accepted, reason = False, "fine-tuned model got worse on answers users approved"
        elif has_wins and wins_evaluation.worst_case_delta < -self.max_case_drop:
            accepted, reason = False, "fine-tuned model broke an answer a user approved"
        else:
            accepted, reason = True, "fine-tuned model improved on held-out failures with no regression"

        report = WeightCycleReport(
            accepted=accepted,
            reason=reason,
            score_before=evaluation.score_before,
            score_after=evaluation.score_after,
            dataset=export,
            holdout_cases=evaluation.scored,
            wins_delta=wins_evaluation.delta if has_wins else None,
            train_score_before=train_evaluation.score_before if train_evaluation.scored else None,
            train_score_after=train_evaluation.score_after if train_evaluation.scored else None,
        )
        self._record(report)
        return report, (candidate_model if accepted else current_model)

    def _compare(self, cases: list[ReplayCase], current_model: Any, candidate_model: Any):
        baseline = self.replay_runner.run(cases, model=current_model)
        candidate = self.replay_runner.run(cases, model=candidate_model)
        return self.evaluator.evaluate(cases, baseline, candidate)

    def _refuse(self, reason: str, *, dataset: Optional[ExportReport] = None) -> WeightCycleReport:
        return WeightCycleReport(
            accepted=False,
            reason=reason,
            score_before=0.0,
            score_after=0.0,
            dataset=dataset,
        )

    def _record(self, report: WeightCycleReport) -> None:
        export = report.dataset
        patch_id = self.db.insert_patch(
            patch_type="model_weights",
            target="model",
            content={
                "description": (
                    f"Fine-tuned weights on {export.examples_written} examples, "
                    f"judged on {report.holdout_cases} held-out failures"
                ),
                "dataset_path": str(export.path),
                "dataset_examples": export.examples_written,
            },
            rationale="Weight-level learning cycle: bake validated experience into the model.",
            source_issue_type="weight_cycle",
        )
        self.db.update_patch_status(
            patch_id=patch_id,
            status="promoted" if report.accepted else "rejected",
            score_before=report.score_before,
            score_after=report.score_after,
            details={
                "reason": report.reason,
                "validation": "holdout",
                "cases_scored": report.holdout_cases,
                "regression_delta": report.wins_delta,
                "train_score_before": report.train_score_before,
                "train_score_after": report.train_score_after,
                "held_out_from_training": export.held_out,
            },
        )
