"""Shared offline fixtures: a tiny in-memory player DB and a scripted fake LLM."""

from __future__ import annotations

import sqlite3
import sys
import uuid
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chat.agent import PlayerDataAgent  # noqa: E402
from chat.config import ChatSettings  # noqa: E402
from chat.db import PlayerDatabase  # noqa: E402

# Mirrors the real schema in csv/data/sample(1).sqlite.
SCHEMA_SQL = """
CREATE TABLE account (
    account_id text,
    created_time timestamp,
    created_device text,
    created_platform text,
    country_code text,
    created_app_store_id int
);
CREATE TABLE account_date_session (
    account_id text,
    date date,
    session_count int,
    session_duration_sec int
);
CREATE TABLE iap_purchase (
    account_id text,
    created_time timestamp,
    package_id_hash text,
    iap_price_usd_cents int,
    app_store_id int
);
"""

ACCOUNTS = [
    ("1", "2016-01-01 10:00:00.000", "iPhone6,2", "iOS", "GB", 1),
    ("2", "2016-01-02 11:00:00.000", "SM-G900F", "Android", "FR", 2),
    ("3", "2016-02-01 12:00:00.000", "iPhone7,1", "iOS", "US", 1),
    ("4", "2016-02-03 09:30:00.000", "SM-G920F", "Android", "US", 2),
]
SESSIONS = [
    ("1", "2016-01-01", 3, 600),
    ("1", "2016-01-02", 1, 120),
    ("2", "2016-01-02", 2, 300),
    ("3", "2016-02-01", 5, 1800),
    ("4", "2016-02-03", 1, 60),
]
PURCHASES = [
    ("1", "2016-01-01 10:30:00.000", "pkgA", 499, 1),
    ("3", "2016-02-01 12:30:00.000", "pkgB", 999, 1),
    ("3", "2016-02-02 08:00:00.000", "pkgA", 499, 1),
]

TABLES = {"account", "account_date_session", "iap_purchase"}


def populate(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.executemany("INSERT INTO account VALUES (?, ?, ?, ?, ?, ?)", ACCOUNTS)
    conn.executemany("INSERT INTO account_date_session VALUES (?, ?, ?, ?)", SESSIONS)
    conn.executemany("INSERT INTO iap_purchase VALUES (?, ?, ?, ?, ?)", PURCHASES)
    # Enough rows to exercise the row cap.
    conn.executemany(
        "INSERT INTO account_date_session VALUES (?, ?, ?, ?)",
        [(f"bulk{i}", "2016-03-01", 1, 10) for i in range(50)],
    )
    conn.commit()


@pytest.fixture
def memory_db():
    """A named, shared-cache in-memory DB; each query opens a fresh guarded connection."""
    uri = f"file:players_{uuid.uuid4().hex}?mode=memory&cache=shared"
    keeper = sqlite3.connect(uri, uri=True)
    populate(keeper)
    db = PlayerDatabase(
        lambda: sqlite3.connect(uri, uri=True, check_same_thread=False),
        query_timeout_sec=5.0,
    )
    yield db
    keeper.close()


@pytest.fixture
def file_db_path(tmp_path: Path) -> Path:
    """An on-disk copy with parentheses in the name, like the real file."""
    path = tmp_path / "sample(1).sqlite"
    conn = sqlite3.connect(path)
    populate(conn)
    conn.close()
    return path


@pytest.fixture
def settings() -> ChatSettings:
    return ChatSettings(model="fake", max_rows=10, query_timeout_sec=5.0)


@pytest.fixture
def make_agent(memory_db, settings):
    def _make(*responses: str) -> PlayerDataAgent:
        llm = FakeListChatModel(responses=list(responses))
        return PlayerDataAgent(memory_db, llm=llm, settings=settings)

    return _make
