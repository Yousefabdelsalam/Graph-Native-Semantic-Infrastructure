"""Fixtures for tests against a real Neo4j.

These tests DELETE EVERYTHING in the database they use, so they only run against
``NEO4J_TEST_URI`` (default bolt://localhost:7688, the tmpfs container):

    docker compose --profile test up -d --wait neo4j-test
    pytest -m integration
"""

from __future__ import annotations

import os

import pytest

from backend.graph.repository import GraphRepository, GraphUnavailableError
from tests.conftest import make_settings


def _test_uri(main_uri: str) -> str:
    uri = os.environ.get("NEO4J_TEST_URI", "bolt://localhost:7688")
    normalise = lambda value: value.replace("127.0.0.1", "localhost").rstrip("/")  # noqa: E731
    if normalise(uri) == normalise(main_uri):
        pytest.exit("NEO4J_TEST_URI points at the application database; refusing to wipe it.")
    return uri


@pytest.fixture(scope="session")
def neo4j_settings(tmp_path_factory: pytest.TempPathFactory):
    settings = make_settings(tmp_path_factory.mktemp("neo4j"))
    return settings.model_copy(update={"neo4j_uri": _test_uri(settings.neo4j_uri)})


@pytest.fixture()
def repository(neo4j_settings):
    repo = GraphRepository(neo4j_settings)
    try:
        repo.connect()
    except GraphUnavailableError as exc:
        pytest.skip(f"Test Neo4j is not running: {exc}")
    repo.clear()
    yield repo
    repo.clear()
    repo.close()
