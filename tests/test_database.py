"""SQLite metadata schema (application-owned)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from database.migrations import apply_migrations
from database.models import Document, IngestionRun, Query, QuerySource
from database.session import Database


def _document(file_hash: str = "a" * 64) -> Document:
    return Document(
        original_filename="notes.txt",
        stored_filename=f"{file_hash[:20]}.txt",
        file_type="txt",
        file_path="/tmp/notes.txt",
        file_hash=file_hash,
        file_size=10,
    )


@pytest.fixture()
def db(tmp_path: Path):
    database = Database(tmp_path / "meta.db")
    yield database
    database.dispose()


def test_schema_is_created_once(db: Database) -> None:
    tables = set(inspect(db.engine).get_table_names())
    assert {"documents", "ingestion_runs", "queries", "query_sources"} <= tables
    assert apply_migrations(db.engine) == []  # idempotent


def test_file_hash_is_unique(db: Database) -> None:
    with db.session() as session:
        session.add(_document())
    with pytest.raises(IntegrityError):
        with db.session() as session:
            session.add(_document())


def test_foreign_keys_are_enforced(db: Database) -> None:
    with pytest.raises(IntegrityError):
        with db.session() as session:
            session.add(IngestionRun(document_id=999, status="x", extraction_method="regex"))


def test_deleting_a_document_cascades(db: Database) -> None:
    with db.session() as session:
        document = _document()
        document.runs.append(IngestionRun(status="processed", extraction_method="regex"))
        query = Query(question="q", retrieval_mode="vector", execution_time=0.1)
        query.sources.append(
            QuerySource(document=document, source_type="chunk", evidence_reference="c1")
        )
        session.add_all([document, query])
    with db.session() as session:
        session.delete(session.scalars(select(Document)).one())
    with db.session() as session:
        assert session.scalars(select(IngestionRun)).all() == []
        assert session.scalars(select(QuerySource)).all() == []
        assert len(session.scalars(select(Query)).all()) == 1
