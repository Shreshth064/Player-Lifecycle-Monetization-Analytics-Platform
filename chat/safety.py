"""SQL safety validation: allow exactly one read-only SELECT, nothing else.

Validation works on the parsed syntax tree (sqlglot), not on string matching,
so keywords inside string literals or comments neither trigger false alarms
nor slip through. The query that is finally executed is *re-generated from
the validated tree* (with comments dropped), so what runs is exactly what was
checked.

This is the first of three layers; see ``chat.db`` for the read-only
connection and the SQLite authorizer that back it up at execution time.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from chat.errors import UnsafeQueryError

MAX_SQL_LENGTH = 5000

# Statement / clause node types that must never appear anywhere in the tree,
# including nested inside CTEs or subqueries (e.g. ``WITH d AS (DELETE ...)``).
_FORBIDDEN_NODE_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Drop",
    "Alter",
    "Create",
    "Pragma",
    "Attach",
    "Detach",
    "Command",  # sqlglot's fallback for anything it can't model (REPLACE, VACUUM, ...)
    "Transaction",
    "Commit",
    "Rollback",
    "Merge",
    "Analyze",
    "Set",
    "Use",
    "Into",  # SELECT ... INTO
    "Copy",
    "LoadData",
)
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name) for name in _FORBIDDEN_NODE_NAMES if hasattr(exp, name)
)

# Functions that can touch the filesystem, load code, or are otherwise not
# needed for analytics. Python's sqlite3 already disables most of these; this
# is defence in depth.
FORBIDDEN_FUNCTIONS = frozenset(
    {
        "load_extension",
        "readfile",
        "writefile",
        "edit",
        "fts3_tokenizer",
        "zipfile",
        "sqlar_compress",
        "sqlar_uncompress",
        "sqlite_compileoption_get",
        "sqlite_compileoption_used",
        "randomblob",
        "zeroblob",
    }
)

_TRAILING_TERMINATORS = re.compile(r"[\s;]+$")


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def _cte_names(tree: exp.Expression) -> set[str]:
    return {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}


def _check_tables(tree: exp.Expression, allowed_tables: set[str]) -> None:
    visible = allowed_tables | _cte_names(tree)
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            # Table-valued functions such as pragma_table_info(...) / json_each(...)
            raise UnsafeQueryError("Table-valued functions are not allowed.")
        if table.args.get("db") is not None or table.args.get("catalog") is not None:
            raise UnsafeQueryError("Schema-qualified table names are not allowed.")
        if table.name.lower() not in visible:
            raise UnsafeQueryError(
                f"Unknown table '{table.name}'. Only the player tables can be queried."
            )


def _enforce_limit(tree: exp.Expression, max_rows: int) -> None:
    """Make sure the outermost query returns at most ``max_rows`` rows."""
    limit = tree.args.get("limit")
    if isinstance(limit, exp.Limit):
        value = limit.expression
        if (
            isinstance(value, exp.Literal)
            and not value.is_string
            and value.is_int
            and 0 <= int(value.this) <= max_rows
        ):
            return  # already a safe, literal limit
    # Missing, too large, negative (unbounded in SQLite) or non-literal: clamp.
    tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))


def validate_select(
    sql: str,
    allowed_tables: Iterable[str],
    max_rows: int,
) -> str:
    """Validate ``sql`` and return a normalised, row-limited SELECT to execute.

    Raises ``UnsafeQueryError`` if the query is anything other than a single
    read-only SELECT over the known tables.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise UnsafeQueryError("No SQL query was produced.")
    if len(sql) > MAX_SQL_LENGTH:
        raise UnsafeQueryError("The SQL query is too long.")
    if max_rows < 1:
        raise ValueError("max_rows must be positive")

    # A single trailing terminator is harmless; any *other* semicolon shows up
    # below as a second statement and is rejected.
    cleaned = _TRAILING_TERMINATORS.sub("", sql.strip())

    try:
        statements = [s for s in sqlglot.parse(cleaned, read="sqlite") if s is not None]
    except ParseError:
        raise UnsafeQueryError("The SQL query could not be parsed.") from None

    if len(statements) != 1:
        raise UnsafeQueryError("Only a single SQL statement is allowed.")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        raise UnsafeQueryError(
            f"Only SELECT queries are allowed (got {tree.key.upper()})."
        )

    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise UnsafeQueryError(
                f"{node.key.upper()} is not allowed; queries must be read-only."
            )
        if isinstance(node, exp.Func) and _function_name(node) in FORBIDDEN_FUNCTIONS:
            raise UnsafeQueryError(f"Function '{_function_name(node)}' is not allowed.")

    _check_tables(tree, {t.lower() for t in allowed_tables})
    _enforce_limit(tree, max_rows)

    return tree.sql(dialect="sqlite", comments=False)
