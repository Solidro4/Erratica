"""Where the record lives.

A store keeps two things: an append-mostly list of records by kind (runs,
cases, verdicts, installed changes) and a small key-value state. Anything
with these five methods can be a store, so the record can live in memory, in
one SQLite file, or in whatever your team already uses.

    add(kind, data) -> id        put(kind, id, data)        rows(kind)
    get_state(key, default)      set_state(key, value)

``persistent`` says whether records outlive the process; the loop uses it to
insist that recorded inputs survive a round trip through JSON.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


class MemoryStore:
    """Keeps everything in this process. Nothing survives a restart."""

    persistent = False

    def __init__(self) -> None:
        self._rows: Dict[str, Dict[int, Any]] = {}
        self._state: Dict[str, Any] = {}
        self._next = 1
        self._lock = threading.Lock()

    def add(self, kind: str, data: Any) -> int:
        with self._lock:
            row_id, self._next = self._next, self._next + 1
            self._rows.setdefault(kind, {})[row_id] = data
            return row_id

    def put(self, kind: str, row_id: int, data: Any) -> None:
        with self._lock:
            self._rows.setdefault(kind, {})[row_id] = data

    def rows(self, kind: str) -> List[Tuple[int, Any]]:
        with self._lock:
            return sorted(self._rows.get(kind, {}).items())

    def get_state(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        self._state[key] = value

    def close(self) -> None:
        pass


class SqliteStore:
    """One SQLite file. Records are stored as JSON, so anyone can read them."""

    persistent = True

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._lock, self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS records ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, "
                "data TEXT NOT NULL, created REAL NOT NULL)"
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS records_kind ON records (kind, id)")
            self._db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    @staticmethod
    def _dump(data: Any) -> str:
        # Values that are not JSON (an output object from some framework,
        # say) are stored as their text, so recording never raises.
        return json.dumps(data, default=str)

    def add(self, kind: str, data: Any) -> int:
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO records (kind, data, created) VALUES (?, ?, ?)",
                (kind, self._dump(data), time.time()),
            )
            return int(cursor.lastrowid)

    def put(self, kind: str, row_id: int, data: Any) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE records SET data = ? WHERE id = ? AND kind = ?",
                (self._dump(data), row_id, kind),
            )

    def rows(self, kind: str) -> List[Tuple[int, Any]]:
        with self._lock:
            found = self._db.execute(
                "SELECT id, data FROM records WHERE kind = ? ORDER BY id", (kind,)
            ).fetchall()
        return [(int(row_id), json.loads(data)) for row_id, data in found]

    def get_state(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._db.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def set_state(self, key: str, value: Any) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, self._dump(value)),
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()


def open_store(store: Any) -> Any:
    """None -> memory; a path -> SQLite; anything else is used as given."""
    if store is None:
        return MemoryStore()
    if isinstance(store, (str, Path)):
        return SqliteStore(store)
    return store
