"""Read-only access to the player SQLite database.

Execution-time safety layers (the parser in ``chat.safety`` is the first):

1. The file is opened through a ``file:...?mode=ro`` URI, so SQLite itself
   refuses any write.
2. ``PRAGMA query_only = ON`` is set on every connection.
3. A SQLite authorizer allows only SELECT, reads of the known player tables
   and non-dangerous functions; every other action is denied by the engine.
4. A progress handler aborts queries that exceed a time budget, and results
   are fetched with ``fetchmany(max_rows)`` as a final row cap.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chat.config import find_database
from chat.errors import DatabaseUnavailableError, QueryExecutionError, UnsafeQueryError
from chat.safety import FORBIDDEN_FUNCTIONS

logger = logging.getLogger(__name__)

ConnectionFactory = Callable[[], sqlite3.Connection]

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}

# SQLite error prefixes that are useful to the user and never contain paths.
_USER_VISIBLE_ERRORS = (
    "no such column",
    "no such table",
    "no such function",
    "ambiguous column",
    "misuse of",
    "near ",
    "wrong number of arguments",
    "incomplete input",
)


def readonly_uri(path: str | os.PathLike[str]) -> str:
    """Build a ``file:`` URI that opens ``path`` read-only.

    ``Path.as_uri`` percent-encodes characters such as ``?`` and ``#``; the
    parentheses in ``sample(1).sqlite`` are valid URI characters.
    """
    return f"{Path(path).resolve().as_uri()}?mode=ro"


def connect_readonly(path: str | os.PathLike[str]) -> sqlite3.Connection:
    return sqlite3.connect(readonly_uri(path), uri=True, check_same_thread=False)


@dataclass(frozen=True)
class Column:
    name: str
    type: str


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    truncated: bool


class PlayerDatabase:
    """Executes validated SELECT statements on a guarded, read-only connection."""

    def __init__(
        self,
        connect: ConnectionFactory,
        *,
        query_timeout_sec: float = 10.0,
    ) -> None:
        self._connect = connect
        self.query_timeout_sec = query_timeout_sec
        self._schema: dict[str, list[Column]] | None = None

    @classmethod
    def from_path(
        cls, path: str | os.PathLike[str] | None = None, **kwargs: Any
    ) -> "PlayerDatabase":
        db_path = find_database(path)
        return cls(lambda: connect_readonly(db_path), **kwargs)

    # -- schema ---------------------------------------------------------------

    def schema(self) -> dict[str, list[Column]]:
        """Return ``{table: [Column, ...]}`` for every user table."""
        if self._schema is None:
            conn = self._open()
            try:
                tables = [
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                    )
                ]
                schema: dict[str, list[Column]] = {}
                for table in tables:
                    info = conn.execute(
                        "SELECT name, type FROM pragma_table_info(?)", (table,)
                    ).fetchall()
                    schema[table] = [Column(name, col_type or "") for name, col_type in info]
            finally:
                conn.close()
            self._schema = schema
        return self._schema

    @property
    def tables(self) -> set[str]:
        return set(self.schema())

    def describe_schema(self) -> str:
        """Render the schema as compact text for the LLM prompt."""
        lines = []
        for table, columns in self.schema().items():
            cols = ", ".join(f"{c.name} {c.type}".strip() for c in columns)
            lines.append(f"- {table}({cols})")
        return "\n".join(lines)

    # -- execution ------------------------------------------------------------

    def execute(self, sql: str, max_rows: int) -> QueryResult:
        """Run an already-validated SELECT on a guarded connection."""
        allowed_tables = {t.lower() for t in self.tables}
        conn = self._open()
        try:
            self._guard(conn, allowed_tables)
            cursor = conn.execute(sql)
            columns = _unique_columns([d[0] for d in cursor.description or []])
            fetched = cursor.fetchmany(max_rows + 1)
        except sqlite3.DatabaseError as exc:
            raise _translate(exc) from None
        finally:
            conn.close()

        truncated = len(fetched) > max_rows
        rows = [dict(zip(columns, row)) for row in fetched[:max_rows]]
        return QueryResult(columns=columns, rows=rows, truncated=truncated)

    def _open(self) -> sqlite3.Connection:
        try:
            return self._connect()
        except sqlite3.Error:
            logger.exception("Could not open player database")
            raise DatabaseUnavailableError("The player database is not available.") from None

    def _guard(self, conn: sqlite3.Connection, allowed_tables: set[str]) -> None:
        conn.execute("PRAGMA query_only = ON")

        def authorizer(
            action: int, arg1: str | None, arg2: str | None, db_name: str | None, *_: Any
        ) -> int:
            if action not in _ALLOWED_ACTIONS:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ and db_name:
                # Real tables report their database ("main"); CTEs/subqueries report "".
                # Only the player tables in "main" may be read: no sqlite_master,
                # no temp or attached databases.
                if db_name != "main" or (arg1 or "").lower() not in allowed_tables:
                    return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in FORBIDDEN_FUNCTIONS:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(authorizer)

        deadline = time.monotonic() + self.query_timeout_sec
        # A non-zero return value interrupts the running statement.
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)


def _unique_columns(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 1
            out.append(name)
    return out


def _translate(exc: sqlite3.DatabaseError) -> Exception:
    message = str(exc)
    lowered = message.lower()
    if any(m in lowered for m in ("not authorized", "prohibited", "readonly", "read-only")):
        return UnsafeQueryError("The query attempted a non read-only operation.")
    if "interrupted" in lowered:
        return QueryExecutionError("The query took too long and was stopped.")
    logger.warning("Query failed: %s", message)
    if lowered.startswith(_USER_VISIBLE_ERRORS):
        return QueryExecutionError(f"The query failed: {message}")
    return QueryExecutionError("The query failed.")
