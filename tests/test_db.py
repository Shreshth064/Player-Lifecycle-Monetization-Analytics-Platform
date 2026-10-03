"""Execution-layer safety: read-only connection, authorizer, row cap, timeout.

These tests bypass the parser on purpose to prove the database layer refuses
writes on its own (defence in depth).
"""

from __future__ import annotations

import sqlite3

import pytest

from chat.config import find_database
from chat.db import PlayerDatabase, connect_readonly, readonly_uri
from chat.errors import DatabaseUnavailableError, QueryExecutionError, UnsafeQueryError
from tests.conftest import TABLES


def test_schema_introspection(memory_db):
    schema = memory_db.schema()
    assert set(schema) == TABLES
    assert [c.name for c in schema["iap_purchase"]] == [
        "account_id",
        "created_time",
        "package_id_hash",
        "iap_price_usd_cents",
        "app_store_id",
    ]
    text = memory_db.describe_schema()
    assert "account_date_session(account_id text, date date" in text.lower()


def test_execute_returns_rows_as_dicts(memory_db):
    result = memory_db.execute(
        "SELECT country_code, COUNT(*) AS players FROM account "
        "GROUP BY country_code ORDER BY country_code",
        max_rows=100,
    )
    assert result.columns == ["country_code", "players"]
    assert result.rows[0] == {"country_code": "FR", "players": 1}
    assert result.truncated is False


def test_row_cap_enforced_even_without_sql_limit(memory_db):
    result = memory_db.execute("SELECT * FROM account_date_session", max_rows=7)
    assert len(result.rows) == 7
    assert result.truncated is True


def test_duplicate_column_names_are_disambiguated(memory_db):
    result = memory_db.execute(
        "SELECT a.account_id, p.account_id FROM account a "
        "JOIN iap_purchase p ON a.account_id = p.account_id LIMIT 1",
        max_rows=10,
    )
    assert result.columns == ["account_id", "account_id_2"]


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO account(account_id) VALUES ('x')",
        "UPDATE account SET country_code = 'XX'",
        "DELETE FROM account",
        "DROP TABLE account",
        "CREATE TABLE evil (a int)",
        "PRAGMA query_only = OFF",
        "ATTACH DATABASE ':memory:' AS evil",
        "SELECT * FROM sqlite_master",
    ],
)
def test_authorizer_blocks_non_reads_on_raw_sql(memory_db, sql):
    with pytest.raises(UnsafeQueryError):
        memory_db.execute(sql, max_rows=10)
    # And nothing changed.
    assert memory_db.execute("SELECT COUNT(*) AS n FROM account", 10).rows == [{"n": 4}]


def test_multiple_statements_refused_by_driver(memory_db):
    with pytest.raises((UnsafeQueryError, QueryExecutionError, sqlite3.ProgrammingError)):
        memory_db.execute("SELECT 1; DELETE FROM account", max_rows=10)
    assert memory_db.execute("SELECT COUNT(*) AS n FROM account", 10).rows == [{"n": 4}]


def test_file_connection_is_opened_read_only(file_db_path):
    uri = readonly_uri(file_db_path)
    assert uri.startswith("file:") and uri.endswith("?mode=ro")

    # Even an *unguarded* connection from connect_readonly cannot write.
    conn = connect_readonly(file_db_path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM account")
    finally:
        conn.close()


def test_file_db_end_to_end_with_parentheses_in_name(file_db_path):
    db = PlayerDatabase.from_path(file_db_path)
    assert db.tables == TABLES
    assert db.execute("SELECT COUNT(*) AS n FROM account", 10).rows == [{"n": 4}]
    with pytest.raises(UnsafeQueryError):
        db.execute("DELETE FROM account", 10)


def test_query_timeout(memory_db):
    memory_db.query_timeout_sec = 0.0
    with pytest.raises(QueryExecutionError, match="too long"):
        memory_db.execute(
            "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) "
            "SELECT COUNT(*) FROM n",
            max_rows=10,
        )


def test_sql_errors_are_user_safe(memory_db):
    with pytest.raises(QueryExecutionError, match="no such column"):
        memory_db.execute("SELECT nope FROM account", 10)


def test_find_database_env_override(file_db_path, monkeypatch):
    monkeypatch.setenv("PLAYER_DB_PATH", str(file_db_path))
    assert find_database() == file_db_path.resolve()


def test_find_database_missing_path_hides_location(monkeypatch, tmp_path):
    secret = tmp_path / "secret-location" / "nope.sqlite"
    monkeypatch.setenv("PLAYER_DB_PATH", str(secret))
    with pytest.raises(DatabaseUnavailableError) as info:
        find_database()
    assert "secret-location" not in str(info.value)


def test_unopenable_database_is_reported_cleanly():
    def broken() -> sqlite3.Connection:
        raise sqlite3.OperationalError("unable to open database file /private/path")

    db = PlayerDatabase(broken)
    with pytest.raises(DatabaseUnavailableError) as info:
        db.schema()
    assert "/private/path" not in str(info.value)


def test_ctes_and_subqueries_are_allowed_by_authorizer(memory_db):
    result = memory_db.execute(
        "WITH payers AS (SELECT DISTINCT account_id FROM iap_purchase) "
        "SELECT COUNT(*) AS n FROM (SELECT * FROM payers)",
        max_rows=10,
    )
    assert result.rows == [{"n": 2}]
