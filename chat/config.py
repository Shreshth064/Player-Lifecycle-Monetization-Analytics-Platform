"""Runtime configuration for the chat agent, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from chat.errors import DatabaseUnavailableError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_DIR = PROJECT_ROOT / "csv" / "data"
DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_MAX_ROWS = 1000
DEFAULT_QUERY_TIMEOUT_SEC = 10.0


def find_database(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate the SQLite database.

    Order: explicit argument, ``PLAYER_DB_PATH`` env var, then the first
    ``*.sqlite`` file under ``csv/data/`` (the shipped file name contains
    parentheses, e.g. ``sample(1).sqlite``, so it is globbed rather than
    hard-coded).
    """
    candidate = explicit or os.getenv("PLAYER_DB_PATH")
    if candidate:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
        raise DatabaseUnavailableError("The player database is not available.")

    matches = sorted(DEFAULT_DB_DIR.glob("*.sqlite")) + sorted(DEFAULT_DB_DIR.glob("*.db"))
    if not matches:
        raise DatabaseUnavailableError("The player database is not available.")
    return matches[0].resolve()


@dataclass(frozen=True)
class ChatSettings:
    model: str = DEFAULT_MODEL
    max_rows: int = DEFAULT_MAX_ROWS
    query_timeout_sec: float = DEFAULT_QUERY_TIMEOUT_SEC

    @classmethod
    def from_env(cls) -> "ChatSettings":
        return cls(
            model=os.getenv("CHAT_MODEL", DEFAULT_MODEL),
            max_rows=int(os.getenv("CHAT_MAX_ROWS", DEFAULT_MAX_ROWS)),
            query_timeout_sec=float(
                os.getenv("CHAT_QUERY_TIMEOUT_SEC", DEFAULT_QUERY_TIMEOUT_SEC)
            ),
        )
