"""Held-out validation splits.

A patch is written from some failures and must be judged on others. If it is
judged on the cases it was written from, it can win by memorising them, and
"validated" stops meaning anything.

Cases are grouped by prompt first, so the same prompt can never sit on both
sides of the split, and the split is deterministic so a learning cycle can be
reproduced from the database.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

from erratica.textutil import normalize_prompt

T = TypeVar("T")

HOLDOUT = "holdout"
IN_SAMPLE = "in_sample"

# With fewer distinct prompts than this there is nothing to hold out.
MIN_DISTINCT_FOR_HOLDOUT = 3


@dataclass
class Split:
    train: list
    holdout: list
    mode: str

    @property
    def is_holdout(self) -> bool:
        return self.mode == HOLDOUT


def _prompt_of(case: Any) -> str:
    if isinstance(case, dict):
        return str(case.get("prompt", ""))
    return str(getattr(case, "prompt", ""))


def _stable_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def distinct_prompts(cases: Sequence[Any], prompt_of: Callable[[Any], str] = _prompt_of) -> int:
    return len({normalize_prompt(prompt_of(case)) for case in cases})


def split_cases(
    cases: Sequence[T],
    *,
    holdout_fraction: float = 0.34,
    min_distinct: int = MIN_DISTINCT_FOR_HOLDOUT,
    prompt_of: Callable[[Any], str] = _prompt_of,
) -> Split:
    """Split cases into the ones a patch may learn from and the ones it is judged on.

    Returns mode ``in_sample`` (train == holdout == all cases) when there are
    too few distinct prompts to hold any out. Callers must record the mode:
    an in-sample result is weak evidence and has to be labelled as such.
    """
    groups: dict[str, list[T]] = {}
    for case in cases:
        groups.setdefault(normalize_prompt(prompt_of(case)), []).append(case)

    if len(groups) < min_distinct:
        everything = list(cases)
        return Split(train=everything, holdout=everything, mode=IN_SAMPLE)

    ordered = sorted(groups, key=_stable_key)
    holdout_count = max(1, min(len(ordered) - 1, math.ceil(len(ordered) * holdout_fraction)))
    holdout_keys = set(ordered[:holdout_count])

    train: list[T] = []
    holdout: list[T] = []
    for key, members in groups.items():
        (holdout if key in holdout_keys else train).extend(members)
    return Split(train=train, holdout=holdout, mode=HOLDOUT)
