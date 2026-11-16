"""The composable core: a loop whose every stage can be replaced.

    from erratica.core import Loop, Change, ReplayGate

See ``erratica.core.loop`` for the two ways to use it.
"""

from erratica.core import checks, proposers
from erratica.core.adapters import from_adapter, ollama, prompted, render_prompt
from erratica.core.gates import CertifiedGate, Paired, ReplayGate, Trial, all_of
from erratica.core.loop import Loop
from erratica.core.proposers import Context
from erratica.core.store import MemoryStore, SqliteStore
from erratica.core.types import APPROVED, FAILURE, HOLDOUT, IN_SAMPLE, AuditFinding, Case, Change, Run, Verdict

__all__ = [
    "APPROVED",
    "AuditFinding",
    "Case",
    "CertifiedGate",
    "Change",
    "Context",
    "FAILURE",
    "HOLDOUT",
    "IN_SAMPLE",
    "Loop",
    "MemoryStore",
    "Paired",
    "ReplayGate",
    "Run",
    "SqliteStore",
    "Trial",
    "Verdict",
    "all_of",
    "checks",
    "from_adapter",
    "ollama",
    "prompted",
    "proposers",
    "render_prompt",
]
