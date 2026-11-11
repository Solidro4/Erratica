from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from erratica import CommandTrainer, RuleAwareDemoModel

from tests.helpers import EngineTestCase, report_failure

PROMPTS = [
    "Customer says order #41327 was returned two weeks ago and the refund still has not arrived.",
    "Customer says order #55210 arrived damaged and asks for a replacement.",
    "Customer says order #77110 has had no tracking updates for five days and asks what happens next.",
    "Customer says they were charged twice for the same subscription renewal and wants it reversed.",
    "Customer reports the shipment for order #90417 is delayed again and asks for an update.",
    "Customer was promised a refund on order #28844 ten days ago and has heard nothing since.",
]


class GeneralisingModel:
    """A fine-tune that actually learned the behavior: good on any support prompt."""

    def __init__(self) -> None:
        self._inner = RuleAwareDemoModel()

    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        if route != "support":
            return self._inner.generate(prompt, system_prompt, route)
        return self._inner.generate(prompt, system_prompt + "\nGive a numbered resolution.", route)


class MemorisingModel:
    """A fine-tune that only memorised its training file."""

    def __init__(self, dataset_path: Path) -> None:
        self.answers: dict[str, str] = {}
        for line in Path(dataset_path).read_text(encoding="utf-8").splitlines():
            messages = {m["role"]: m["content"] for m in json.loads(line)["messages"]}
            self.answers[messages["user"]] = messages["assistant"]

    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        return self.answers.get(prompt, "Received.")


class ForgetfulModel(GeneralisingModel):
    """Learned support replies but forgot how to do everything else."""

    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        if route != "support":
            return "ok"
        return super().generate(prompt, system_prompt, route)


class WeightLoopTests(EngineTestCase):
    def _seed(self, ai, prompts=PROMPTS) -> None:
        teacher = GeneralisingModel()
        for prompt in prompts:
            report_failure(
                ai,
                prompt,
                issue_type="missing_resolution",
                ideal_response=teacher.generate(prompt, "", "support"),
            )

    def test_fine_tune_that_generalises_is_promoted_and_swapped_in(self) -> None:
        ai = self.engine(enable_recall=False)
        self._seed(ai)
        original = ai.model

        report = ai.run_weight_cycle(lambda path: GeneralisingModel(), dataset_path=self.tmp / "sft.jsonl")

        self.assertTrue(report.accepted, report.reason)
        self.assertGreater(report.score_after, report.score_before)
        self.assertGreaterEqual(report.holdout_cases, 2)
        self.assertIsInstance(ai.model, GeneralisingModel)
        self.assertIsNot(ai.model, original)
        self.assertIs(ai.replay_runner.model, ai.model)

        card = ai.model_card(model_name="tuned")
        self.assertIn("model_weights", card)
        self.assertIn("Fine-tuned weights on", card)
        self.assertIn("held-out", card)

    def test_held_out_prompts_are_kept_out_of_the_training_file(self) -> None:
        ai = self.engine(enable_recall=False)
        self._seed(ai)
        seen: dict[str, set] = {}

        def trainer(path: Path):
            seen["prompts"] = {
                json.loads(line)["messages"][1]["content"]
                for line in Path(path).read_text(encoding="utf-8").splitlines()
            }
            return GeneralisingModel()

        report = ai.run_weight_cycle(trainer, dataset_path=self.tmp / "sft.jsonl")

        self.assertGreaterEqual(report.dataset.held_out, 2)
        self.assertEqual(len(seen["prompts"]) + report.dataset.held_out, len(PROMPTS))
        self.assertTrue(seen["prompts"] < set(PROMPTS))

    def test_fine_tune_that_only_memorised_is_rejected(self) -> None:
        ai = self.engine(enable_recall=False)
        self._seed(ai)
        original = ai.model

        report = ai.run_weight_cycle(MemorisingModel, dataset_path=self.tmp / "sft.jsonl")

        self.assertFalse(report.accepted)
        self.assertIn("held-out", report.reason)
        self.assertIs(ai.model, original)
        # It looks perfect on what it was trained on, which is exactly the trap.
        self.assertGreater(report.train_score_after, 0.9)
        self.assertLess(report.score_after, 0.3)
        self.assertEqual(ai.db.list_patch_history(limit=1)[0]["status"], "rejected")

    def test_fine_tune_that_forgets_approved_answers_is_rejected(self) -> None:
        ai = self.engine(enable_recall=False)
        self._seed(ai)
        liked = ai.generate("Explain what a refund is in one line.", route="general")
        ai.approve(liked.interaction_id)

        report = ai.run_weight_cycle(lambda path: ForgetfulModel(), dataset_path=self.tmp / "sft.jsonl")

        self.assertFalse(report.accepted)
        self.assertIn("approved", report.reason)
        self.assertLess(report.wins_delta, 0)
        self.assertGreater(report.delta, 0.5)

    def test_too_few_failures_refuses_to_run(self) -> None:
        ai = self.engine(enable_recall=False)
        self._seed(ai, PROMPTS[:2])
        original = ai.model
        called = []

        report = ai.run_weight_cycle(
            lambda path: called.append(path) or GeneralisingModel(),
            dataset_path=self.tmp / "sft.jsonl",
        )

        self.assertFalse(report.accepted)
        self.assertIn("not enough distinct failures", report.reason)
        self.assertEqual(called, [])
        self.assertIs(ai.model, original)

    def test_no_experience_is_a_no_op(self) -> None:
        ai = self.engine()
        report = ai.run_weight_cycle(lambda path: GeneralisingModel(), dataset_path=self.tmp / "sft.jsonl")
        self.assertFalse(report.accepted)


class CommandTrainerTests(EngineTestCase):
    _COPY = "import pathlib, sys; pathlib.Path(sys.argv[2]).write_bytes(pathlib.Path(sys.argv[1]).read_bytes())"

    def test_runs_the_command_and_loads_its_output(self) -> None:
        dataset = self.tmp / "data.jsonl"
        dataset.write_text('{"a": 1}\n', encoding="utf-8")
        output = self.tmp / "model.bin"
        trainer = CommandTrainer(
            [sys.executable, "-c", self._COPY, "{dataset}", "{output}"],
            output_path=output,
            loader=lambda path: ("loaded", Path(path).read_text(encoding="utf-8")),
        )

        self.assertEqual(trainer.train(dataset), ("loaded", '{"a": 1}\n'))

    def test_missing_output_and_failed_command_raise(self) -> None:
        dataset = self.tmp / "data.jsonl"
        dataset.write_text("{}\n", encoding="utf-8")
        silent = CommandTrainer(
            [sys.executable, "-c", "pass"],
            output_path=self.tmp / "never.bin",
            loader=lambda path: path,
        )
        with self.assertRaises(RuntimeError):
            silent.train(dataset)

        failing = CommandTrainer(
            [sys.executable, "-c", "raise SystemExit(3)"],
            output_path=self.tmp / "never.bin",
            loader=lambda path: path,
        )
        with self.assertRaises(subprocess.CalledProcessError):
            failing.train(dataset)

    def test_plugs_into_the_weight_cycle(self) -> None:
        ai = self.engine(enable_recall=False)
        teacher = GeneralisingModel()
        for prompt in PROMPTS:
            report_failure(
                ai,
                prompt,
                issue_type="missing_resolution",
                ideal_response=teacher.generate(prompt, "", "support"),
            )
        trainer = CommandTrainer(
            [sys.executable, "-c", self._COPY, "{dataset}", "{output}"],
            output_path=self.tmp / "tuned.bin",
            loader=lambda path: GeneralisingModel(),
        )

        report = ai.run_weight_cycle(trainer, dataset_path=self.tmp / "sft.jsonl")

        self.assertTrue(report.accepted, report.reason)
        self.assertTrue((self.tmp / "tuned.bin").exists())
