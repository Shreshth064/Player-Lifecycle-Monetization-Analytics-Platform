"""Exceptions raised by the chat package.

Every message here is safe to show to an end user: no file paths, keys or
stack traces. Internal details are logged, never attached to these errors.
"""

from __future__ import annotations


class ChatError(Exception):
    """Base class for user-facing chat errors."""

    code = "chat_error"


class UnsafeQueryError(ChatError):
    """The generated SQL is not a single, read-only SELECT statement."""

    code = "unsafe_query"


class QueryExecutionError(ChatError):
    """A validated query failed or timed out while running."""

    code = "query_failed"


class DatabaseUnavailableError(ChatError):
    """The player database could not be located or opened."""

    code = "database_unavailable"


class LLMUnavailableError(ChatError):
    """The language model is not configured or did not respond."""

    code = "llm_unavailable"
