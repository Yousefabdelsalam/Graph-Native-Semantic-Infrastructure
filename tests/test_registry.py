"""Document registration, duplicate prevention and run bookkeeping."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.config import load_settings
from backend.services.registry import DocumentRegistry
from database.models import DocumentStatus
from database.session import Database


@pytest.fixture()
def registry(tmp_path: Path):
    settings = load_settings(data_dir=tmp_path)
    database = Database(settings.metadata_db)
    yield DocumentRegistry(database, settings)
    database.dispose()


def test_identical_content_is_registered_once(registry: DocumentRegistry) -> None:
    first = registry.register("a.txt", b"VisionCore depends on Redis.")
    again = registry.register("renamed.txt", b"VisionCore depends on Redis.")
    other = registry.register("a.txt", b"VisionCore depends on Kafka.")
    assert first.created and other.created and not again.created
    assert again.document.id == first.document.id
    assert registry.counts()["documents"] == 2
    assert Path(first.document.file_path).read_bytes() == b"VisionCore depends on Redis."


def test_run_lifecycle_updates_document(registry: DocumentRegistry) -> None:
    document = registry.register("a.txt", b"x").document
    failed = registry.start_run(document.id, "regex")
    registry.finish_run(failed, duration=0.1, error="boom")
    assert registry.get(document.id).processing_status == DocumentStatus.FAILED
    assert registry.get(document.id).error_message == "boom"

    ok = registry.start_run(document.id, "regex")
    registry.finish_run(ok, duration=0.2, entities=3, relationships=2)
    stored = registry.get(document.id)
    assert stored.processing_status == DocumentStatus.PROCESSED
    assert stored.error_message is None and stored.processed_at is not None
    assert [run.status for run in stored.runs] == ["failed", "processed"]


def test_query_log_ignores_unknown_documents(registry: DocumentRegistry) -> None:
    document = registry.register("a.txt", b"x").document
    registry.log_query(
        "q",
        "vector",
        "answer",
        0.01,
        [
            {"document_id": document.id, "source_type": "chunk", "evidence_reference": "c1"},
            {"document_id": 999, "source_type": "chunk", "evidence_reference": "c2"},
        ],
    )
    assert registry.counts()["queries"] == 1
