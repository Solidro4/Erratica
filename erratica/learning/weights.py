from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from erratica.learning.evaluator import PatchEvaluator
from erratica.learning.exporter import ExportReport, TrainingDataExporter
from erratica.learning.replay import ReplayCase, ReplayDatasetBuilder, ReplayRunner
from erratica.storage.db import Database


class ModelTrainer(Protocol):
    """Anything that turns a JSONL dataset into a new candidate model.

    Implementations typically shell out to a LoRA/QLoRA trainer (unsloth,
    axolotl, llama.cpp) and return a model adapter loading the tuned weights.
    """

    def train(self, dataset_path: Path) -> Any:
        ...


@dataclass
class WeightCycleReport:
    accepted: bool
    reason: str
    score_before: float
    score_after: float
    dataset: ExportReport | None = None

    @property
    def delta(self) -> float:
        return self.score_after - self.score_before


class WeightLoop:
    """Closes the loop from validated experience to model weights.

    The same contract as every other Erratica patch applies to the weights:
    a fine-tuned candidate model must beat the current model on the replay
    set before it is allowed to take over. A bad fine-tune is rejected and
    the current model stays in place, so the loop can only move forward.
    """

    def __init__(
        self,
        db: Database,
        exporter: TrainingDataExporter,
        dataset_builder: ReplayDatasetBuilder,
        evaluator: PatchEvaluator,
        policy: Any,
        router: Any,
        *,
        min_delta: float = 0.05,
    ) -> None:
        self.db = db
        self.exporter = exporter
        self.dataset_builder = dataset_builder
        self.evaluator = evaluator
        self.policy = policy
        self.router = router
        self.min_delta = min_delta

    def run(
        self,
        current_model: Any,
        trainer: ModelTrainer | Callable[[Path], Any],
        *,
        dataset_path: str | Path = Path("data") / "erratica_sft.jsonl",
        replay_limit: int = 200,
    ) -> tuple[WeightCycleReport, Any]:
        """Returns the cycle report and the model to use going forward."""
        cases = self._replay_cases(replay_limit)
        if not cases:
            report = WeightCycleReport(
                accepted=False,
                reason="no recorded failures to verify a fine-tune against",
                score_before=0.0,
                score_after=0.0,
            )
            return report, current_model

        export = self.exporter.export(dataset_path)
        if export.examples_written == 0:
            report = WeightCycleReport(
                accepted=False,
                reason="no validated experience to fine-tune on yet",
                score_before=0.0,
                score_after=0.0,
                dataset=export,
            )
            return report, current_model

        train = trainer.train if hasattr(trainer, "train") else trainer
        candidate_model = train(Path(dataset_path))

        baseline_results = ReplayRunner(current_model, self.policy, self.router).run(cases)
        candidate_results = ReplayRunner(candidate_model, self.policy, self.router).run(cases)
        evaluation = self.evaluator.evaluate(cases, baseline_results, candidate_results)

        accepted = evaluation.delta > self.min_delta
        reason = (
            "fine-tuned model outperformed current model on replay"
            if accepted
            else "fine-tuned model did not outperform current model on replay"
        )
        self._record(export, evaluation.score_before, evaluation.score_after, accepted)

        report = WeightCycleReport(
            accepted=accepted,
            reason=reason,
            score_before=evaluation.score_before,
            score_after=evaluation.score_after,
            dataset=export,
        )
        return report, (candidate_model if accepted else current_model)

    def _replay_cases(self, limit: int) -> list[ReplayCase]:
        failures = self.db.list_failed_interactions(limit=limit)
        return self.dataset_builder.build(failures)

    def _record(
        self,
        export: ExportReport,
        score_before: float,
        score_after: float,
        accepted: bool,
    ) -> None:
        patch_id = self.db.insert_patch(
            patch_type="model_weights",
            target="model",
            content={
                "description": (
                    f"Fine-tuned weights on {export.examples_written} validated examples"
                ),
                "dataset_path": str(export.path),
                "dataset_examples": export.examples_written,
            },
            rationale="Weight-level learning cycle: bake validated experience into the model.",
            source_issue_type="weight_cycle",
        )
        self.db.update_patch_status(
            patch_id=patch_id,
            status="promoted" if accepted else "rejected",
            score_before=score_before,
            score_after=score_after,
        )
