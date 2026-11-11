"""Restricted execution of learned tools.

A learned tool is a small Python function ``run(prompt: str) -> str`` that a
proposer model wrote to compute something the answering model keeps getting
wrong (arithmetic, parsing, date maths). Its output is given to the model as
a verified note.

Two layers keep a tool contained:

1. Static screening (``validate_tool_source``): the source may only import a
   short allow-list of pure-computation modules and may not touch dunder
   names, private attributes, files, or dynamic execution.
2. Process isolation (``run_tools``): tools run in a separate interpreter
   with a restricted set of builtins, a wall-clock timeout, and a cap on
   output size.

This is defence in depth for code you are about to review, not a security
boundary against a determined attacker. Tools are off unless you pass
``enable_tools=True``, every promoted tool is stored in the audit trail, and
you should read a tool before trusting a deployment that uses it.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from typing import Optional

ALLOWED_MODULES = (
    "math",
    "re",
    "json",
    "statistics",
    "datetime",
    "decimal",
    "fractions",
    "itertools",
    "collections",
    "string",
    "calendar",
)

_FORBIDDEN_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "breakpoint",
    "exit",
    "quit",
    "help",
    "memoryview",
    "type",
    "object",
    "super",
    "classmethod",
    "staticmethod",
}

MAX_SOURCE_CHARS = 6000
MAX_OUTPUT_CHARS = 600
DEFAULT_TIMEOUT = 5.0


class ToolValidationError(ValueError):
    pass


def validate_tool_source(source: str) -> None:
    """Raise ToolValidationError unless the source passes static screening."""
    if len(source) > MAX_SOURCE_CHARS:
        raise ToolValidationError(f"tool source is longer than {MAX_SOURCE_CHARS} characters")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ToolValidationError(f"tool source does not parse: {exc.msg}") from exc

    defines_run = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    raise ToolValidationError(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                raise ToolValidationError(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in _FORBIDDEN_NAMES:
                raise ToolValidationError(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ToolValidationError(f"access to private attribute '{node.attr}' is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise ToolValidationError("string literals containing '__' are not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.ClassDef)):
            raise ToolValidationError(f"{type(node).__name__} is not allowed in a tool")
        elif isinstance(node, ast.FunctionDef) and node.name == "run":
            if node in tree.body:
                defines_run = True
    if not defines_run:
        raise ToolValidationError("tool must define a top-level function run(prompt)")


# The runner executes inside the child interpreter. It reads one JSON job from
# stdin and writes one JSON result to stdout.
_RUNNER = r"""
import builtins, json, sys

ALLOWED = set(json.loads(sys.argv[1]))
MAX_OUT = int(sys.argv[2])
job = json.loads(sys.stdin.buffer.read().decode("utf-8"))

_real_import = builtins.__import__

def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED:
        raise ImportError("import of %r is not allowed" % name)
    return _real_import(name, globals, locals, fromlist, level)

SAFE = {}
for _name in (
    "abs all any bool dict divmod enumerate filter float frozenset int isinstance len list "
    "map max min pow range repr reversed round set sorted str sum tuple zip chr ord "
    "ArithmeticError Exception IndexError KeyError LookupError OverflowError "
    "StopIteration TypeError ValueError ZeroDivisionError True False None"
).split():
    if hasattr(builtins, _name):
        SAFE[_name] = getattr(builtins, _name)
SAFE["__import__"] = _import
SAFE["print"] = lambda *args, **kwargs: None

results = []
for tool in job["tools"]:
    entry = {"name": tool["name"], "output": "", "error": None}
    try:
        namespace = {"__builtins__": SAFE, "__name__": "erratica_tool"}
        exec(compile(tool["source"], "<tool:%s>" % tool["name"], "exec"), namespace)
        value = namespace["run"](job["prompt"])
        entry["output"] = "" if value is None else str(value)[:MAX_OUT]
    except BaseException as exc:
        entry["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
    results.append(entry)

sys.stdout.write(json.dumps(results))
"""


def run_tools(
    tools: list[dict[str, str]],
    prompt: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, Optional[str]]]:
    """Run each tool on the prompt in one isolated child process.

    ``tools`` is a list of ``{"name": ..., "source": ...}``. Returns one
    ``{"name", "output", "error"}`` entry per tool. A tool that fails static
    screening, raises, or times out yields an error and an empty output; it
    never raises into the caller.
    """
    if not tools:
        return []

    runnable: list[dict[str, str]] = []
    results: dict[str, dict[str, Optional[str]]] = {}
    for tool in tools:
        try:
            validate_tool_source(tool["source"])
        except ToolValidationError as exc:
            results[tool["name"]] = {"name": tool["name"], "output": "", "error": f"rejected: {exc}"}
        else:
            runnable.append({"name": tool["name"], "source": tool["source"]})

    if runnable:
        job = json.dumps({"tools": runnable, "prompt": prompt})
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    _RUNNER,
                    json.dumps(list(ALLOWED_MODULES)),
                    str(MAX_OUTPUT_CHARS),
                ],
                input=job.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            parsed = json.loads(completed.stdout.decode("utf-8"))
            for entry in parsed:
                results[entry["name"]] = entry
        except subprocess.TimeoutExpired:
            for tool in runnable:
                results[tool["name"]] = {
                    "name": tool["name"],
                    "output": "",
                    "error": f"timed out after {timeout:.0f}s",
                }
        except (ValueError, OSError) as exc:
            for tool in runnable:
                results[tool["name"]] = {
                    "name": tool["name"],
                    "output": "",
                    "error": f"sandbox failure: {exc}",
                }

    return [results[tool["name"]] for tool in tools]


def render_tool_notes(results: list[dict[str, Optional[str]]]) -> str:
    """Format successful tool outputs as a block for the system prompt."""
    lines = [
        f"- {entry['name']}: {str(entry['output']).strip()}"
        for entry in results
        if not entry.get("error") and str(entry.get("output") or "").strip()
    ]
    if not lines:
        return ""
    return "\n".join(
        ["Verified tool results (computed for this request; prefer them over your own estimates):"]
        + lines
    )
