import pytest
from fastapi.testclient import TestClient

from scanner.app import app


@pytest.fixture(scope="session")
def client():
    """One app for the whole run: the MCP session manager starts once per process."""
    with TestClient(app) as c:
        yield c
