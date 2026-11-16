"""The nouns of the composable core.

Three rules keep the core malleable:

- **Your system is any callable** ``input -> output``. Erratica never needs
  to know whether it is one prompt, a chain, a crew of agents, or a search
  pipeline.
- **Cases are data.** A mistake that happened (``failure``) or an answer
  someone approved (``approved``), with what was wanted, as plain values.
- **Changes are data.** A change is a list of operations on a configuration
  (set a value, append to a list, remove from a list). Because it is data it
  can be stored, shown in a diff, replayed, and taken out again.

Code lives in three replaceable places only: ``build(config)``, which turns a
configuration into a system; checks, which score an answer; and gates, which
decide whether a change may be installed.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from erratica.textutil import normalize_prompt

FAILURE = "failure"
APPROVED = "approved"

HOLDOUT = "holdout"      # judged on failures the change was not written from
IN_SAMPLE = "in_sample"  # judged on failures its author could see: weak evidence

System = Callable[[Any], Any]


def key_of(value: Any) -> str:
    """A stable text key for an input, used to tell whether two are the same."""
    if isinstance(value, str):
        return normalize_prompt(value)
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)


@dataclass
class Run:
    """One answer the system gave."""

    input: Any
    output: Any
    version: int = 0
    id: Optional[int] = None


@dataclass
class Case:
    """A recorded input and what is known about a good answer to it.

    ``kind`` is ``"failure"`` (the system got this wrong) or ``"approved"``
    (someone accepted the answer; later changes must not break it).
    ``issue`` is a short label shared by failures of the same kind; failures
    are grouped by it. Everything else is optional evidence for the checks:
    ``expect`` (the answer that was wanted), ``must`` / ``must_not`` (terms a
    good answer does or does not contain), and ``meta`` for your own checks.
    """

    input: Any
    kind: str = FAILURE
    issue: str = "general"
    output: Any = None
    expect: Any = None
    must: Tuple[str, ...] = ()
    must_not: Tuple[str, ...] = ()
    note: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    version: int = 0
    id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in (FAILURE, APPROVED):
            raise ValueError(f"kind must be '{FAILURE}' or '{APPROVED}', not {self.kind!r}")
        self.must = tuple(self.must or ())
        self.must_not = tuple(self.must_not or ())

    @property
    def key(self) -> str:
        return key_of(self.input)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["must"], data["must_not"] = list(self.must), list(self.must_not)
        data.pop("id")
        return data

    @classmethod
    def from_dict(cls, data: dict, id: Optional[int] = None) -> "Case":
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data and name != "id"}
        return cls(id=id, **known)


SET, APPEND, REMOVE = "set", "append", "remove"


def _walk(config: Any, path: str, create: bool) -> Tuple[Any, str]:
    """The container holding the last part of a dotted path, and that part."""
    parts = [part for part in str(path).split(".") if part]
    if not parts:
        raise ValueError("a change needs a non-empty path")
    node = config
    for part in parts[:-1]:
        if not isinstance(node, dict):
            raise TypeError(f"cannot follow path '{path}': '{part}' is not inside a mapping")
        if part not in node:
            if not create:
                return None, parts[-1]
            node[part] = {}
        node = node[part]
    if not isinstance(node, dict):
        raise TypeError(f"cannot follow path '{path}': its parent is not a mapping")
    return node, parts[-1]


@dataclass(frozen=True)
class Change:
    """A named, replayable edit to a configuration.

    Build one with ``Change.set``, ``Change.append`` or ``Change.remove``, and
    join several with ``+``. Paths are dotted (``"agent.rules"``).
    """

    name: str
    ops: Tuple[dict, ...]
    kind: str = "custom"
    issue: Optional[str] = None
    note: str = ""

    @classmethod
    def set(cls, path: str, value: Any, *, name: Optional[str] = None, **details: Any) -> "Change":
        return cls(name or f"set {path}", ({"op": SET, "path": path, "value": value},), **details)

    @classmethod
    def append(cls, path: str, value: Any, *, name: Optional[str] = None, **details: Any) -> "Change":
        return cls(name or f"append to {path}", ({"op": APPEND, "path": path, "value": value},), **details)

    @classmethod
    def remove(cls, path: str, value: Any, *, name: Optional[str] = None, **details: Any) -> "Change":
        return cls(name or f"remove from {path}", ({"op": REMOVE, "path": path, "value": value},), **details)

    def __add__(self, other: "Change") -> "Change":
        if not isinstance(other, Change):
            return NotImplemented
        return Change(
            f"{self.name} + {other.name}",
            self.ops + other.ops,
            kind=self.kind if self.kind == other.kind else "custom",
            issue=self.issue or other.issue,
            note=" ".join(part for part in (self.note, other.note) if part),
        )

    def apply(self, config: Any) -> Any:
        """A new configuration with this change applied. The input is not modified."""
        result = copy.deepcopy(config)
        for op in self.ops:
            kind, path, value = op["op"], op["path"], copy.deepcopy(op.get("value"))
            node, last = _walk(result, path, create=kind != REMOVE)
            if kind == SET:
                node[last] = value
            elif kind == APPEND:
                target = node.setdefault(last, [])
                if not isinstance(target, list):
                    raise TypeError(f"cannot append to '{path}': it is not a list")
                target.append(value)
            elif kind == REMOVE:
                target = None if node is None else node.get(last)
                if isinstance(target, list) and value in target:
                    target.remove(value)
            else:
                raise ValueError(f"unknown operation {kind!r}")
        return result

    def with_issue(self, issue: Optional[str]) -> "Change":
        if self.issue is not None or issue is None:
            return self
        return Change(self.name, self.ops, kind=self.kind, issue=issue, note=self.note)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ops": [dict(op) for op in self.ops],
            "kind": self.kind,
            "issue": self.issue,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Change":
        return cls(
            data["name"],
            tuple(dict(op) for op in data.get("ops", ())),
            kind=data.get("kind", "custom"),
            issue=data.get("issue"),
            note=data.get("note", ""),
        )


@dataclass
class Verdict:
    """What a gate decided about one candidate, and the evidence for it."""

    accepted: bool
    reason: str = ""
    change: Optional[str] = None
    issue: Optional[str] = None
    gate: str = ""
    # "holdout" or "in_sample": see the constants above.
    validation: str = IN_SAMPLE
    # Mean score on the target failures before and after, and how many were scored.
    before: Optional[float] = None
    after: Optional[float] = None
    cases: int = 0
    # Chance of a gain this large on these cases if the change did nothing.
    p_value: Optional[float] = None
    # Largest mean drop over the guarded groups (approved answers, other failures).
    guard_loss: Optional[float] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    installed: bool = False
    version: Optional[int] = None
    id: Optional[int] = None

    def __bool__(self) -> bool:
        return self.accepted

    @property
    def gain(self) -> Optional[float]:
        if self.before is None or self.after is None:
            return None
        return self.after - self.before

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("id")
        return data


@dataclass
class AuditFinding:
    """What switching one installed change off did to the scores."""

    change: str
    # Mean score with the change minus without it. At or below zero means
    # the change is no longer earning its place.
    cost_of_removal: Optional[float]
    cases: int
    removed: bool
    reason: str = ""


Check = Callable[[Case, Any], Optional[float]]
Events = Dict[str, List[Callable[[Any], None]]]
