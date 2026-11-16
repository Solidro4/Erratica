"""The examples that need no model must keep running and saying what the docs say."""

from __future__ import annotations

import contextlib
import io
import runpy
import unittest
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def output_of(name: str) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return buffer.getvalue()


class ExampleTests(unittest.TestCase):
    def test_every_example_is_valid_python(self):
        # The examples that need a model or another framework cannot run
        # here, but a syntax error in one should still fail the build.
        for path in sorted(EXAMPLES.glob("*.py")):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_quickstart(self):
        text = output_of("quickstart.py")
        self.assertIn("rejected   shortcut: lowers the score on approved answers", text)
        self.assertIn("INSTALLED  rule: Cite every claim with a [source] marker.", text)
        self.assertIn("judged on 2 holdout cases: 0.00 -> 1.00", text)
        self.assertIn("after rollback: {'rules': []}", text)

    def test_any_system(self):
        text = output_of("any_system.py")
        self.assertIn("rejected   match word stems: still misses 1 of 2 target queries", text)
        self.assertIn("INSTALLED  match word stems + return one more document", text)
        self.assertIn("after restart: {'top_k': 2, 'stemming': True} | score: {'score': 1.0, 'cases': 5}", text)


if __name__ == "__main__":
    unittest.main()
