"""Retry policy for transient LLM failures (HTTP 429 / 503).

Only rate limiting (429 / ResourceExhausted) and temporary unavailability
(503 / ServiceUnavailable) are retried, with exponential backoff and jitter.
Everything else -- auth errors, 400 bad requests, other 5xx, and our own
ChatErrors such as SQL-safety rejections -- fails immediately.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from langchain_core.exceptions import ModelRateLimitError
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from chat.errors import ChatError

logger = logging.getLogger(__name__)

T = TypeVar("T")

TRANSIENT_STATUS_CODES = frozenset({429, 503})
# google.api_core exception names, for older / alternative Google clients.
TRANSIENT_ERROR_NAMES = frozenset({"ResourceExhausted", "ServiceUnavailable", "TooManyRequests"})


def _causes(exc: BaseException, max_depth: int = 5):
    """Yield ``exc`` and the exceptions it was raised from."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(seen) < max_depth:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transient_llm_error(exc: BaseException) -> bool:
    """True for 429 / 503-style failures that are worth retrying."""
    if isinstance(exc, ChatError):
        return False
    for err in _causes(exc):
        if isinstance(err, ModelRateLimitError):
            return True
        if type(err).__name__ in TRANSIENT_ERROR_NAMES:
            return True
        code = getattr(err, "code", None)
        if code is None:
            code = getattr(err, "status_code", None)
        try:
            if code is not None and int(code) in TRANSIENT_STATUS_CODES:
                return True
        except (TypeError, ValueError):
            pass
    return False


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    initial_wait_sec: float = 1.0,
    max_wait_sec: float = 10.0,
) -> T:
    """Call ``fn``, retrying transient LLM errors with exponential backoff + jitter."""
    retrying = Retrying(
        retry=retry_if_exception(is_transient_llm_error),
        stop=stop_after_attempt(max(1, attempts)),
        wait=wait_exponential_jitter(
            initial=initial_wait_sec, max=max_wait_sec, jitter=initial_wait_sec
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    return retrying(fn)
