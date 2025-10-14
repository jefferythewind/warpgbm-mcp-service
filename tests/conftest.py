"""
Pytest configuration for local testing.
Uses local_dev app instead of app.main to get local GPU support.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def test_app():
    """
    Create test app using local_dev for local GPU support.
    Falls back to app.main if local_dev fails.
    Module scope = one app per test file.
    """
    try:
        # Try to use local_dev (has GPU functions injected)
        import local_dev
        print("✅ Using local_dev app (with local GPU support)")
        return local_dev.app  # Return the FastAPI app instance
    except Exception as e:
        # Fall back to regular app.main
        print(f"⚠️  local_dev failed, falling back to app.main: {e}")
        import app.main
        return app.main.app


@pytest.fixture(scope="function")
def client(test_app):
    """
    Create fresh TestClient for each test to avoid rate limiting issues.
    Function scope = new client per test.
    """
    return TestClient(test_app)

