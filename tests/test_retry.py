"""Retry/backoff for transient LLM errors (offline: errors are raised by a fake LLM).

The exceptions are the real classes langchain-google-genai raises for HTTP
429 / 5xx / 4xx responses, so the predicate is tested against what
production code actually sees.
"""

from __future__ import annotations

from typing import Any

import pytest
from google.genai.errors import ClientError
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_google_genai.chat_models import (
    GoogleAPIError,
    GoogleAuthenticationError,
    GoogleInvalidRequestError,
    GoogleRateLimitError,
)

from chat.agent import PlayerDataAgent
from chat.errors import LLMUnavailableError, UnsafeQueryError
from chat.retry import call_with_retry, is_transient_llm_error


def rate_limited() -> Exception:
    return GoogleRateLimitError("Error calling model (RESOURCE_EXHAUSTED): 429 quota")


def unavailable() -> Exception:
    return GoogleAPIError(code=503, response_json={"error": {"code": 503, "status": "UNAVAILABLE"}})


def internal_error() -> Exception:
    return GoogleAPIError(code=500, response_json={"error": {"code": 500, "status": "INTERNAL"}})


def auth_error() -> Exception:
    return GoogleAuthenticationError("Error calling model (UNAUTHENTICATED): 401 API key invalid")


def bad_request() -> Exception:
    return GoogleInvalidRequestError("Error calling model (INVALID_ARGUMENT): 400")


class ScriptedLLM(FakeListChatModel):
    """Fake chat model: each call pops the next script item; exceptions are raised."""

    script: list[Any]
    calls: int = 0

    def _call(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def agent_with(memory_db, settings, *script: Any) -> tuple[PlayerDataAgent, ScriptedLLM]:
    llm = ScriptedLLM(responses=[""], script=list(script))
    return PlayerDataAgent(memory_db, llm=llm, settings=settings), llm


# --- classification ------------------------------------------------------------


class ResourceExhausted(Exception):
    """Stand-in for google.api_core.exceptions.ResourceExhausted."""


class ServiceUnavailable(Exception):
    """Stand-in for google.api_core.exceptions.ServiceUnavailable."""


class HTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    "exc",
    [
        rate_limited(),
        unavailable(),
        ResourceExhausted("quota"),
        ServiceUnavailable("down"),
        HTTPError(429),
        HTTPError(503),
        ClientError(429, {"error": {"code": 429}}),
    ],
)
def test_transient_errors_are_retryable(exc):
    assert is_transient_llm_error(exc)


def test_transient_cause_is_detected_through_wrapping():
    try:
        try:
            raise ClientError(429, {"error": {"code": 429}})
        except ClientError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert is_transient_llm_error(outer)


@pytest.mark.parametrize(
    "exc",
    [
        auth_error(),
        bad_request(),
        internal_error(),  # 500 is not in the 429/503 policy
        HTTPError(400),
        HTTPError(401),
        ValueError("bad"),
        UnsafeQueryError("Only SELECT queries are allowed."),
        LLMUnavailableError("not configured"),
    ],
)
def test_non_transient_errors_are_not_retryable(exc):
    assert not is_transient_llm_error(exc)


# --- call_with_retry -------------------------------------------------------------


def test_call_with_retry_uses_backoff_between_attempts(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("tenacity.nap.time.sleep", sleeps.append)
    outcomes: list[Any] = [rate_limited(), unavailable(), "ok"]

    def flaky() -> str:
        item = outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    assert call_with_retry(flaky, attempts=3, initial_wait_sec=1.0, max_wait_sec=10.0) == "ok"
    assert len(sleeps) == 2
    # Exponential: ~1s then ~2s, each plus up to 1s of jitter.
    assert 1.0 <= sleeps[0] <= 2.0
    assert 2.0 <= sleeps[1] <= 3.0


def test_call_with_retry_respects_max_wait(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("tenacity.nap.time.sleep", sleeps.append)

    def always_busy() -> str:
        raise unavailable()

    with pytest.raises(GoogleAPIError):
        call_with_retry(always_busy, attempts=5, initial_wait_sec=4.0, max_wait_sec=5.0)
    assert len(sleeps) == 4
    assert all(s <= 5.0 for s in sleeps)


# --- agent integration -------------------------------------------------------------


@pytest.mark.parametrize("transient", [rate_limited, unavailable])
def test_agent_retries_transient_error_then_succeeds(memory_db, settings, transient):
    agent, llm = agent_with(
        memory_db,
        settings,
        transient(),
        "SELECT COUNT(*) AS n FROM account",
        "There are 4 players.",
    )
    result = agent.ask("How many players?")
    assert result.rows == [{"n": 4}]
    assert result.answer == "There are 4 players."
    assert llm.calls == 3  # failed SQL call, retried SQL call, summary call


def test_agent_retries_summary_call_too(memory_db, settings):
    agent, llm = agent_with(
        memory_db,
        settings,
        "SELECT COUNT(*) AS n FROM account",
        rate_limited(),
        unavailable(),
        "There are 4 players.",
    )
    assert agent.ask("How many players?").answer == "There are 4 players."
    assert llm.calls == 4


def test_agent_gives_up_after_max_attempts(memory_db, settings):
    agent, llm = agent_with(memory_db, settings, rate_limited(), rate_limited(), rate_limited(), "unused")
    with pytest.raises(LLMUnavailableError, match="busy or rate-limited"):
        agent.ask("How many players?")
    assert llm.calls == settings.llm_max_attempts == 3


@pytest.mark.parametrize("permanent", [auth_error, bad_request, internal_error])
def test_agent_does_not_retry_non_transient_errors(memory_db, settings, permanent):
    agent, llm = agent_with(memory_db, settings, permanent(), "SELECT 1 FROM account")
    with pytest.raises(LLMUnavailableError, match="did not respond") as info:
        agent.ask("How many players?")
    assert llm.calls == 1
    assert "API key" not in str(info.value)


def test_sql_safety_rejection_is_not_retried(memory_db, settings):
    agent, llm = agent_with(memory_db, settings, "DROP TABLE account", "SELECT 1 FROM account")
    with pytest.raises(UnsafeQueryError):
        agent.ask("Drop everything")
    assert llm.calls == 1
    assert memory_db.execute("SELECT COUNT(*) AS n FROM account", 10).rows == [{"n": 4}]
