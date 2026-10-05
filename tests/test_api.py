"""FastAPI contract tests with the agent dependency overridden (offline)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import api
from chat.agent import PlayerDataAgent
from chat.errors import DatabaseUnavailableError


@pytest.fixture
def client_with(memory_db, settings):
    clients = []

    def _make(*responses: str) -> TestClient:
        agent = PlayerDataAgent(
            memory_db, llm=FakeListChatModel(responses=list(responses)), settings=settings
        )
        api.app.dependency_overrides[api.get_agent] = lambda: agent
        client = TestClient(api.app, raise_server_exceptions=False)
        clients.append(client)
        return client

    yield _make
    api.app.dependency_overrides.clear()


def test_health():
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_documented_shape(client_with):
    client = client_with(
        "SELECT created_platform, COUNT(*) AS players FROM account GROUP BY 1 ORDER BY 1",
        "Android and iOS each have 2 players.",
    )
    response = client.post("/chat", json={"question": "Players by platform?"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"sql", "rows", "answer"}
    assert body["sql"].startswith("SELECT created_platform")
    assert body["rows"] == [
        {"created_platform": "Android", "players": 2},
        {"created_platform": "iOS", "players": 2},
    ]
    assert body["answer"] == "Android and iOS each have 2 players."


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO account(account_id) VALUES ('x')",
        "DROP TABLE account",
        "SELECT * FROM account; DROP TABLE account",
        "PRAGMA table_info(account)",
    ],
)
def test_chat_rejects_unsafe_sql_with_400(client_with, memory_db, sql):
    client = client_with(sql)
    response = client.post("/chat", json={"question": "do something nasty"})
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "unsafe_query"
    assert "Traceback" not in body["detail"]
    assert memory_db.execute("SELECT COUNT(*) AS n FROM account", 10).rows == [{"n": 4}]


def test_chat_row_limit(client_with, settings):
    client = client_with("SELECT * FROM account_date_session", "Many rows.")
    body = client.post("/chat", json={"question": "all sessions"}).json()
    assert len(body["rows"]) == settings.max_rows


def test_chat_unanswerable(client_with):
    client = client_with("UNANSWERABLE: no ad spend data")
    response = client.post("/chat", json={"question": "What is ROAS?"})
    assert response.status_code == 200
    assert response.json()["sql"] == ""
    assert "can't answer" in response.json()["answer"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "   "}, {"question": 123}, {"question": "x" * 501}, {"q": "hi"}],
)
def test_chat_input_validation_returns_clean_400(client_with, payload):
    client = client_with("SELECT 1")
    response = client.post("/chat", json=payload)
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "invalid_request"
    assert set(body) == {"error", "detail"}


def test_malformed_json_returns_400(client_with):
    client = client_with("SELECT 1")
    response = client.post(
        "/chat", content="{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400


def test_unexpected_errors_do_not_leak_internals(memory_db, settings):
    class ExplodingAgent:
        def ask(self, question):
            raise RuntimeError("/home/user/secret/path GOOGLE_API_KEY=abc123")

    api.app.dependency_overrides[api.get_agent] = lambda: ExplodingAgent()
    try:
        client = TestClient(api.app, raise_server_exceptions=False)
        response = client.post("/chat", json={"question": "hi"})
    finally:
        api.app.dependency_overrides.clear()
    assert response.status_code == 500
    assert response.json() == {"error": "internal_error", "detail": "An internal error occurred."}
    assert "abc123" not in response.text and "/home" not in response.text


def test_missing_database_returns_503(monkeypatch):
    def unavailable():
        raise DatabaseUnavailableError("The player database is not available.")

    api.app.dependency_overrides[api.get_agent] = unavailable
    try:
        response = TestClient(api.app).post("/chat", json={"question": "hi"})
    finally:
        api.app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()["error"] == "database_unavailable"


def test_missing_api_key_returns_503(client_with, monkeypatch, memory_db, settings):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    agent = PlayerDataAgent(memory_db, llm=None, settings=settings)
    api.app.dependency_overrides[api.get_agent] = lambda: agent
    response = TestClient(api.app).post("/chat", json={"question": "hi"})
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "llm_unavailable"
    assert "GOOGLE_API_KEY" in body["detail"]  # names the variable, never its value
