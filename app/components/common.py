"""Shared Streamlit resources and small rendering helpers."""

from __future__ import annotations

from typing import Iterable

import streamlit as st

from backend.config import configure_logging, load_settings
from backend.graph.repository import GraphUnavailableError
from backend.retrieval.vector import EmbeddingUnavailableError
from backend.services.lab import KnowledgeLab


@st.cache_resource(show_spinner="Starting the knowledge lab…")
def _create_lab() -> KnowledgeLab:
    settings = load_settings()
    configure_logging(settings.log_level)
    return KnowledgeLab(settings)


def get_lab() -> KnowledgeLab:
    return _create_lab()


def require_graph(lab: KnowledgeLab) -> dict[str, list[dict]]:
    """Load the graph from Neo4j, or stop the page with an actionable message."""
    try:
        return lab.graph()
    except GraphUnavailableError as exc:
        st.error(str(exc))
        st.code("docker compose up -d --wait", language="powershell")
        st.stop()


@st.cache_resource(show_spinner="Loading the embedding model…")
def _warm_vector_index(_lab: KnowledgeLab) -> int:
    return _lab.vector.count()


def require_vectors(lab: KnowledgeLab) -> int:
    try:
        _warm_vector_index(lab)
        return lab.vector.count()
    except EmbeddingUnavailableError as exc:
        st.error(str(exc))
        st.stop()


def table(rows: Iterable[dict], *, empty: str = "Nothing to show yet.", **options) -> None:
    rows = list(rows)
    if not rows:
        st.caption(empty)
        return
    st.dataframe(rows, hide_index=True, **options)


def entity_label(entity: dict) -> str:
    return f"{entity['name']} · {entity['type']}"
