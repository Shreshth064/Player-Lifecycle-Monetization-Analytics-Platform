"""FastAPI backend for "Chat with your player data".

Run with:  uvicorn api:app
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from chat import (
    ChatError,
    DatabaseUnavailableError,
    LLMUnavailableError,
    PlayerDataAgent,
    QueryExecutionError,
    UnsafeQueryError,
)
from chat.agent import MAX_QUESTION_LENGTH

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Player Data Chat API",
    description="Ask natural-language questions about player data. "
    "Queries are read-only, SELECT-only and row-limited.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class ChatResponse(BaseModel):
    sql: str
    rows: list[dict[str, Any]]
    answer: str


class ErrorResponse(BaseModel):
    error: str
    detail: str


_STATUS_BY_ERROR: dict[type[ChatError], int] = {
    UnsafeQueryError: 400,
    QueryExecutionError: 400,
    DatabaseUnavailableError: 503,
    LLMUnavailableError: 503,
}


@lru_cache(maxsize=1)
def get_agent() -> PlayerDataAgent:
    return PlayerDataAgent.from_env()


def _error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "detail": detail})


@app.exception_handler(ChatError)
async def chat_error_handler(_: Request, exc: ChatError) -> JSONResponse:
    # ChatError messages are written to be user-safe (no paths, keys or traces).
    return _error(_STATUS_BY_ERROR.get(type(exc), 400), exc.code, str(exc))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Report which field failed without echoing the submitted input back.
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err.get('loc', ()) if p != 'body') or 'body'}: {err.get('msg', 'invalid')}"
        for err in exc.errors()
    )
    return _error(400, "invalid_request", problems or "Invalid request.")


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error", exc_info=exc)
    return _error(500, "internal_error", "An internal error occurred.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def chat(request: ChatRequest, agent: PlayerDataAgent = Depends(get_agent)) -> ChatResponse:
    try:
        result = agent.ask(request.question)
    except ValueError as exc:
        return _error(400, "invalid_request", str(exc))
    return ChatResponse(sql=result.sql, rows=result.rows, answer=result.answer)
