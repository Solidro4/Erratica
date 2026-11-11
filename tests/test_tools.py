from __future__ import annotations

import re
import unittest

from erratica.tools import ToolValidationError, render_tool_notes, run_tools, validate_tool_source

from tests.helpers import EngineTestCase, report_failure

SUM_TOOL = '''
import re

def run(prompt: str) -> str:
    numbers = [int(value) for value in re.findall(r":\\s*(\\d+)", prompt)]
    if not numbers:
        return ""
    return "total=" + str(sum(numbers))
'''


class SandboxTests(unittest.TestCase):
    def test_valid_tool_passes_screening_and_runs(self) -> None:
        validate_tool_source(SUM_TOOL)
        result = run_tools([{"name": "sum", "source": SUM_TOOL}], "A: 3, B: 4")[0]
        self.assertIsNone(result["error"])
        self.assertEqual(result["output"], "total=7")
        self.assertIn("sum: total=7", render_tool_notes([result]))

    def test_dangerous_sources_are_rejected(self) -> None:
        bad_sources = {
            "import os": "import os\ndef run(prompt):\n    return os.getcwd()",
            "from subprocess": "from subprocess import run as r\ndef run(prompt):\n    return ''",
            "open": "def run(prompt):\n    return open('x').read()",
            "dunder": "def run(prompt):\n    return prompt.__class__.__name__",
            "eval": "def run(prompt):\n    return str(eval(prompt))",
            "dunder string": "def run(prompt):\n    return '{0.__class__}'.format(prompt)",
            "no run": "def helper(prompt):\n    return ''",
            "syntax": "def run(prompt) return ''",
            "class": "class A:\n    pass\ndef run(prompt):\n    return ''",
        }
        for label, source in bad_sources.items():
            with self.subTest(label):
                with self.assertRaises(ToolValidationError):
                    validate_tool_source(source)
                result = run_tools([{"name": "bad", "source": source}], "x")[0]
                self.assertTrue(result["error"])
                self.assertEqual(result["output"], "")

    def test_runtime_error_and_timeout_are_contained(self) -> None:
        crash = "def run(prompt):\n    return str(1 // 0)"
        loop = "def run(prompt):\n    while True:\n        pass"
        crashed, fine = run_tools(
            [{"name": "crash", "source": crash}, {"name": "sum", "source": SUM_TOOL}],
            "A: 1, B: 2",
        )
        self.assertIn("ZeroDivisionError", crashed["error"])
        self.assertEqual(fine["output"], "total=3")

        hung = run_tools([{"name": "loop", "source": loop}], "x", timeout=1.5)[0]
        self.assertIn("timed out", hung["error"])
        self.assertEqual(render_tool_notes([hung, crashed]), "")

    def test_import_outside_allow_list_fails_at_runtime_too(self) -> None:
        # Screening is bypassed here on purpose to check the second layer.
        from erratica.tools import sandbox

        original = sandbox.validate_tool_source
        sandbox.validate_tool_source = lambda source: None
        try:
            source = "import os\ndef run(prompt):\n    return os.getcwd()"
            result = run_tools([{"name": "bad", "source": source}], "x")[0]
        finally:
            sandbox.validate_tool_source = original
        self.assertIn("ImportError", result["error"])


class AdditionModel:
    """Cannot add. States a total only when a tool computed it."""

    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        match = re.search(r"total=(\d+)", system_prompt)
        return f"Total: {match.group(1)}" if match else "Total: roughly a lot"


class ScriptedProposer:
    """Stands in for a proposer model: writes a useless rule and a working tool."""

    def __init__(self, tool_source: str = SUM_TOOL) -> None:
        self.tool_source = tool_source
        self.requests: list[str] = []

    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        self.requests.append(prompt)
        if "def run(prompt: str)" in prompt:
            return f"Here you go:\n```python\n{self.tool_source}\n```"
        return "Add the numbers carefully before answering."


class ToolRungTests(EngineTestCase):
    _SEGMENTS = [(3, 4), (10, 2), (5, 5), (8, 1), (6, 7)]

    def _seed(self, ai) -> None:
        for a, b in self._SEGMENTS:
            report_failure(
                ai,
                f"Revenue by segment. North: {a}, South: {b}. What is the total?",
                route="finance",
                issue_type="wrong_total",
                required_terms=[f"Total: {a + b}"],
            )

    def test_tool_is_promoted_when_rules_cannot_fix_the_failure(self) -> None:
        proposer = ScriptedProposer()
        ai = self.engine(model=AdditionModel(), proposer=proposer, enable_tools=True)
        self._seed(ai)

        decisions = ai.run_learning_cycle()

        self.assertEqual([d.patch_type for d in decisions], ["prompt_rule", "prompt_rule", "tool"])
        self.assertEqual([d.accepted for d in decisions], [False, False, True])
        self.assertEqual(decisions[-1].validation, "holdout")
        self.assertEqual(decisions[-1].score_after, 1.0)
        self.assertEqual(ai.dashboard()["metrics"]["active_tools"], 1)

        # Works on values it never saw.
        answer = ai.generate("Revenue by segment. North: 40, South: 2. What is the total?", route="finance")
        self.assertEqual(answer.text, "Total: 42")

        # And the audit can measure and remove it like any other patch.
        self.assertEqual(ai.run_audit()[0].action, "kept")
        self.assertTrue(ai.rollback(decisions[-1].patch_id))
        answer = ai.generate("Revenue by segment. North: 40, South: 2. What is the total?", route="finance")
        self.assertEqual(answer.text, "Total: roughly a lot")

    def test_buggy_first_draft_is_repaired_from_what_it_did_on_training_cases(self) -> None:
        buggy = SUM_TOOL.replace("sum(numbers)", "max(numbers)")

        class RepairingProposer(ScriptedProposer):
            def generate(self, prompt: str, system_prompt: str, route: str) -> str:
                self.requests.append(prompt)
                if "Your previous attempt" in prompt:
                    return f"```python\n{SUM_TOOL}\n```"
                if "def run(prompt: str)" in prompt:
                    return f"```python\n{buggy}\n```"
                return "Add the numbers carefully before answering."

        proposer = RepairingProposer()
        ai = self.engine(model=AdditionModel(), proposer=proposer, enable_tools=True)
        self._seed(ai)

        tool_decision = ai.run_learning_cycle()[-1]

        self.assertEqual(tool_decision.patch_type, "tool")
        self.assertTrue(tool_decision.accepted)
        repair_request = next(r for r in proposer.requests if "Your previous attempt" in r)
        self.assertIn("Your function returned: total=", repair_request)
        self.assertIn("The assistant then answered: Total:", repair_request)
        self.assertIn("sum(numbers)", ai.db.list_tools()[0]["source"])

    def test_tool_attempts_limits_the_number_of_drafts(self) -> None:
        useless = "def run(prompt: str) -> str:\n    return 'note'"
        proposer = ScriptedProposer(useless)
        ai = self.engine(model=AdditionModel(), proposer=proposer, enable_tools=True, tool_attempts=2)
        self._seed(ai)

        tool_decision = ai.run_learning_cycle()[-1]

        self.assertFalse(tool_decision.accepted)
        drafts = [r for r in proposer.requests if "def run(prompt: str)" in r]
        self.assertEqual(len(drafts), 2)

    def test_tool_rung_is_off_by_default(self) -> None:
        ai = self.engine(model=AdditionModel(), proposer=ScriptedProposer())
        self._seed(ai)

        decisions = ai.run_learning_cycle()

        self.assertNotIn("tool", [d.patch_type for d in decisions])
        self.assertFalse(any(d.accepted for d in decisions))

    def test_unsafe_tool_is_rejected_before_it_runs(self) -> None:
        unsafe = "import os\n\ndef run(prompt: str) -> str:\n    return os.getcwd()"
        ai = self.engine(model=AdditionModel(), proposer=ScriptedProposer(unsafe), enable_tools=True)
        self._seed(ai)

        tool_decision = ai.run_learning_cycle()[-1]

        self.assertEqual(tool_decision.patch_type, "tool")
        self.assertFalse(tool_decision.accepted)
        self.assertIn("static screening", tool_decision.reason)
        self.assertEqual(ai.dashboard()["metrics"]["active_tools"], 0)

    def test_tool_that_does_nothing_is_rejected(self) -> None:
        useless = "def run(prompt: str) -> str:\n    return ''"
        ai = self.engine(model=AdditionModel(), proposer=ScriptedProposer(useless), enable_tools=True)
        self._seed(ai)

        tool_decision = ai.run_learning_cycle()[-1]

        self.assertFalse(tool_decision.accepted)
        self.assertIn("returned nothing", tool_decision.reason)


if __name__ == "__main__":
    unittest.main()
