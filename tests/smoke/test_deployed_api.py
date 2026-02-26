"""Smoke tests for local API app wiring."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


pytestmark = pytest.mark.smoke


def test_root_dashboard_renders_html():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "MerchantMind" in response.text
