"""Parser-level safety: only a single read-only SELECT may pass."""

from __future__ import annotations

import pytest

from chat.errors import UnsafeQueryError
from chat.safety import validate_select
from tests.conftest import TABLES

MAX_ROWS = 1000


def check(sql: str, max_rows: int = MAX_ROWS) -> str:
    return validate_select(sql, TABLES, max_rows)


# --- allowed --------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM account",
        "select country_code, count(*) from account group by 1 order by 2 desc",
        "SELECT a.country_code, SUM(p.iap_price_usd_cents) / 100.0 AS revenue "
        "FROM account a JOIN iap_purchase p ON a.account_id = p.account_id GROUP BY 1",
        "WITH payers AS (SELECT DISTINCT account_id FROM iap_purchase) "
        "SELECT COUNT(*) FROM payers",
        "SELECT account_id FROM account UNION SELECT account_id FROM iap_purchase",
        "SELECT * FROM account WHERE country_code IN "
        "(SELECT country_code FROM account WHERE created_platform = 'iOS')",
        "SELECT COUNT(*) FROM account;",  # single trailing terminator is fine
        "SELECT COUNT(*) FROM account ;  \n",
        "SELECT strftime('%m', created_time) AS m, COUNT(*) FROM account GROUP BY m",
    ],
)
def test_valid_selects_pass(sql):
    out = check(sql)
    assert out.upper().startswith(("SELECT", "WITH"))
    assert "LIMIT" in out.upper()


def test_keywords_inside_string_literals_are_not_false_positives():
    out = check("SELECT * FROM account WHERE country_code = 'DROP TABLE x; DELETE'")
    assert "'DROP TABLE x; DELETE'" in out


def test_output_is_regenerated_from_the_tree_without_comments():
    out = check("SELECT 1 FROM account -- ; DROP TABLE account")
    assert "DROP" not in out.upper()
    assert ";" not in out


# --- rejected statement types ------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO account VALUES ('x', NULL, NULL, NULL, NULL, NULL)",
        "UPDATE account SET country_code = 'XX'",
        "DELETE FROM account",
        "DROP TABLE account",
        "ALTER TABLE account ADD COLUMN pwned int",
        "CREATE TABLE evil (a int)",
        "CREATE VIEW v AS SELECT * FROM account",
        "REPLACE INTO account VALUES ('x', NULL, NULL, NULL, NULL, NULL)",
        "INSERT OR REPLACE INTO account(account_id) VALUES ('x')",
        "ATTACH DATABASE '/tmp/evil.db' AS evil",
        "DETACH DATABASE evil",
        "PRAGMA table_info(account)",
        "PRAGMA writable_schema = ON",
        "VACUUM",
        "BEGIN TRANSACTION",
        "ANALYZE",
    ],
)
def test_non_select_statements_are_rejected(sql):
    with pytest.raises(UnsafeQueryError):
        check(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM account; DROP TABLE account",
        "SELECT * FROM account; DELETE FROM account;",
        "SELECT 1 FROM account; SELECT 2 FROM account",
        "SELECT * FROM account;; INSERT INTO account(account_id) VALUES ('x')",
        "SELECT * FROM account; PRAGMA query_only = OFF",
        "SELECT * FROM account; ATTACH DATABASE 'x.db' AS x",
    ],
)
def test_semicolon_chaining_is_rejected(sql):
    with pytest.raises(UnsafeQueryError, match="single SQL statement"):
        check(sql)


def test_writes_hidden_inside_a_cte_are_rejected():
    with pytest.raises(UnsafeQueryError):
        check("WITH d AS (DELETE FROM account RETURNING *) SELECT * FROM d")


def test_select_into_is_rejected():
    with pytest.raises(UnsafeQueryError):
        check("SELECT * INTO backup FROM account")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT load_extension('/tmp/evil.so')",
        "SELECT readfile('/etc/passwd')",
        "SELECT writefile('/tmp/x', 'data')",
        "SELECT LOAD_EXTENSION('x') FROM account",
        "SELECT zeroblob(1000000000)",
    ],
)
def test_dangerous_functions_are_rejected(sql):
    with pytest.raises(UnsafeQueryError, match="not allowed"):
        check(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM sqlite_master",
        "SELECT * FROM secrets",
        "SELECT * FROM pragma_table_info('account')",
        "SELECT * FROM main.account",
        "SELECT * FROM evil.account",
    ],
)
def test_only_known_tables_can_be_read(sql):
    with pytest.raises(UnsafeQueryError):
        check(sql)


@pytest.mark.parametrize("sql", ["", "   ", ";", None])
def test_empty_input_is_rejected(sql):
    with pytest.raises(UnsafeQueryError):
        check(sql)


def test_unparseable_sql_is_rejected():
    with pytest.raises(UnsafeQueryError):
        check("SELECT FROM WHERE (((")


def test_overlong_sql_is_rejected():
    with pytest.raises(UnsafeQueryError, match="too long"):
        check("SELECT " + ", ".join(["account_id"] * 1000) + " FROM account")


# --- row limit -------------------------------------------------------------------


def test_limit_added_when_missing():
    assert check("SELECT * FROM account").endswith("LIMIT 1000")


def test_small_limit_is_kept():
    assert check("SELECT * FROM account LIMIT 5").endswith("LIMIT 5")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM account LIMIT 1000000",
        "SELECT * FROM account LIMIT -1",  # unbounded in SQLite
        "SELECT * FROM account LIMIT (SELECT 99999)",
        "SELECT * FROM account LIMIT '5000'",
    ],
)
def test_large_negative_or_dynamic_limits_are_clamped(sql):
    assert check(sql).endswith("LIMIT 1000")


def test_limit_keeps_offset():
    out = check("SELECT * FROM account LIMIT 50000 OFFSET 20")
    assert "LIMIT 1000" in out and "OFFSET 20" in out


def test_limit_applies_to_outer_query_of_a_union():
    out = check("SELECT account_id FROM account UNION SELECT account_id FROM iap_purchase")
    assert out.endswith("LIMIT 1000")


def test_limit_applies_to_outer_query_not_just_subquery():
    out = check("SELECT * FROM (SELECT * FROM account LIMIT 99999)")
    assert out.endswith("LIMIT 1000")


def test_invalid_max_rows():
    with pytest.raises(ValueError):
        validate_select("SELECT 1 FROM account", TABLES, 0)
