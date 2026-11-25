"""Shared fixtures.  All data is synthetic and no paid API is ever called.

The first run downloads the local embedding model (about 70 MB) once.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from backend.config import Settings, load_settings
from backend.extraction.extractor import ExtractionPipeline
from backend.ingestion.parser import DocumentProcessor
from backend.schemas import DocumentExtraction, ParsedDocument, Segment
from backend.services.lab import KnowledgeLab
from tests.fakes import InMemoryRepository


def make_settings(data_dir: Path) -> Settings:
    return load_settings(data_dir=data_dir, groq_api_key=SecretStr(""))


def make_document(text: str, document_id: int = 1, filename: str = "test.txt") -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id,
        filename=filename,
        file_type="txt",
        text=text,
        segments=[Segment(label="body", start=0, end=len(text))],
        parser="test",
    )


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return make_settings(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def processor() -> DocumentProcessor:
    return DocumentProcessor()


@pytest.fixture(scope="session")
def pipeline(settings: Settings) -> ExtractionPipeline:
    return ExtractionPipeline(settings, "regex")


@pytest.fixture(scope="session")
def sample_files(settings: Settings) -> list[Path]:
    return sorted(p for p in settings.sample_dir.iterdir() if p.suffix != ".json")


@pytest.fixture(scope="session")
def sample_extractions(
    processor: DocumentProcessor, pipeline: ExtractionPipeline, sample_files: list[Path]
) -> list[DocumentExtraction]:
    return [
        pipeline.extract(processor.parse(path, number, path.name))
        for number, path in enumerate(sample_files, start=1)
    ]


@pytest.fixture()
def lab(tmp_path: Path):
    instance = KnowledgeLab(make_settings(tmp_path), repository=InMemoryRepository())
    yield instance
    instance.close()


@pytest.fixture(scope="session")
def loaded_lab(tmp_path_factory: pytest.TempPathFactory):
    """A lab with the sample corpus ingested.  Treat as read-only."""
    settings = make_settings(tmp_path_factory.mktemp("loaded"))
    instance = KnowledgeLab(settings, repository=InMemoryRepository())
    outcomes = instance.load_sample_data()
    assert [o.status for o in outcomes] == ["processed"] * 6, outcomes
    yield instance
    instance.close()
