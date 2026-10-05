"""Chat with your player data: a read-only, SELECT-only text-to-SQL agent."""

from chat.agent import ChatResult, PlayerDataAgent
from chat.errors import (
    ChatError,
    DatabaseUnavailableError,
    LLMUnavailableError,
    QueryExecutionError,
    UnsafeQueryError,
)

__all__ = [
    "ChatError",
    "ChatResult",
    "DatabaseUnavailableError",
    "LLMUnavailableError",
    "PlayerDataAgent",
    "QueryExecutionError",
    "UnsafeQueryError",
]
