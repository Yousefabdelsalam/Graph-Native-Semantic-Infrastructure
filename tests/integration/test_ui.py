"""Smoke tests: every Streamlit page renders and the main interactions work."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
PAGES = ["overview", "documents", "graph_explorer", "query_playground", "provenance", "evaluation"]


@pytest.fixture()
def app_environment(repository, neo4j_settings, tmp_path, monkeypatch):
    """Point the app at the test database and an empty temporary data directory."""
    monkeypatch.setenv("NEO4J_URI", neo4j_settings.neo4j_uri)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GROQ_API_KEY", "")
    st.cache_resource.clear()
    yield repository
    st.cache_resource.clear()


def _page(name: str) -> AppTest:
    return AppTest.from_file(str(ROOT / "app" / "pages" / f"{name}.py"), default_timeout=90)


@pytest.mark.parametrize("name", PAGES)
def test_page_renders_when_empty(app_environment, name: str) -> None:
    page = _page(name).run()
    assert not page.exception, page.exception


@pytest.fixture()
def populated(app_environment):
    overview = _page("overview").run()
    next(b for b in overview.button if b.label == "Load sample documents").click().run()
    assert app_environment.stats()["entities"] >= 25
    return app_environment


@pytest.mark.parametrize("name", PAGES)
def test_page_renders_with_data(populated, name: str) -> None:
    page = _page(name).run()
    assert not page.exception, page.exception
    assert not page.error, [e.value for e in page.error]


@pytest.mark.parametrize("mode", ["vector", "graph", "hybrid", "semantica"])
def test_query_in_each_mode(populated, mode: str) -> None:
    page = _page("query_playground").run()
    page.text_input(key="question").input("Which projects use PostgreSQL?").run()
    next(s for s in page.selectbox if s.label == "Retrieval mode").set_value(mode).run()
    next(b for b in page.button if b.label == "Run query").click().run()
    assert not page.exception, page.exception
    assert any("Retrieved evidence" in m.value for m in page.markdown)


def test_evaluation_runs_from_the_page(populated) -> None:
    page = _page("evaluation").run()
    next(b for b in page.button if b.label == "Run evaluation").click().run()
    assert not page.exception, page.exception
    assert any(h.value == "Overall" for h in page.subheader)


def test_graph_page_actions(populated) -> None:
    page = _page("graph_explorer").run()
    for label in (
        "Analyse graph",
        "Validate graph against SHACL",
        "Run inference",
        "Infer ontology",
        "Find shortest path",
    ):
        next(b for b in page.button if b.label == label).click().run()
        assert not page.exception, (label, page.exception)


def test_decision_can_be_recorded_and_traced(populated) -> None:
    page = _page("provenance").run()
    next(b for b in page.button if b.label == "Record decision").click().run()
    assert any("Recorded decision" in s.value for s in page.success)
    page.toggle[0].set_value(True).run()
    assert not page.exception, page.exception


def test_unreachable_database_is_explained(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://127.0.0.1:1")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    st.cache_resource.clear()
    page = _page("overview").run()
    assert not page.exception
    assert "Cannot connect to Neo4j" in page.error[0].value
    st.cache_resource.clear()
