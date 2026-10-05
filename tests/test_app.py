"""Smoke tests: every Streamlit dashboard page renders without exceptions."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")

PAGES = [
    "Executive Player Health",
    "Player Segmentation & Monetization",
    "Churn Risk & Monitoring",
    "Chat with your player data",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.header, "page rendered no header"
