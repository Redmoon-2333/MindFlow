"""Tests for packaged React SPA static serving."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.app import SPAStaticFiles


def test_spa_static_files_serves_assets_and_route_fallback(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<main>MindFlow UI</main>", encoding="utf-8")
    (tmp_path / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
    app = FastAPI()
    app.mount("/", SPAStaticFiles(directory=tmp_path, html=True), name="frontend")
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert "MindFlow UI" in client.get("/panel").text
    assert client.get("/assets/app.js").text == "console.log('ok')"
    assert client.get("/missing.js").status_code == 404
    api_missing = client.get("/api/v1/missing")
    assert api_missing.status_code == 404
    assert "MindFlow UI" not in api_missing.text
