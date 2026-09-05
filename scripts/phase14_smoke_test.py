"""Verify the API and SQLite persistence path without starting a server."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402
from storage import engine  # noqa: E402


def main() -> None:
    assert app.health()["status"] == "ok"
    app.initialize_database()
    routes = {route.path for route in app.app.routes}
    required = {"/health", "/candidates/parse", "/candidates", "/rank", "/lookalikes/{candidate_name}"}
    assert required.issubset(routes)
    print("Phase 14-18 smoke test passed: API routes and persistence configuration loaded.")
    print(f"Routes: {len(routes)}; database engine: {engine.url.get_backend_name()}")


if __name__ == "__main__":
    main()
