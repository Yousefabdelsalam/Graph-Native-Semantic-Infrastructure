"""Ingestion, parsing, normalisation and chunking through Semantica."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.ingestion.parser import DocumentParsingError, DocumentProcessor
from backend.extraction.sentences import split_sentences


def _parse(processor: DocumentProcessor, sample_files: list[Path], suffix: str):
    path = next(p for p in sample_files if p.suffix == suffix)
    return processor.parse(path, 1, path.name)


def test_pdf_keeps_page_boundaries(processor, sample_files) -> None:
    document = _parse(processor, sample_files, ".pdf")
    assert [s.label for s in document.segments] == ["page 1", "page 2"]
    second = document.segments[1]
    assert "DocuMind" in document.text[second.start : second.end]
    assert document.metadata["page_count"] == "2"


def test_docx_keeps_section_headings(processor, sample_files) -> None:
    document = _parse(processor, sample_files, ".docx")
    assert "section 'Datastores'" in [s.label for s in document.segments]
    assert "VisionCore depends on PostgreSQL." in document.text


def test_html_headings_stay_on_their_own_line(processor, sample_files) -> None:
    document = _parse(processor, sample_files, ".html")
    assert "Project Atlas\nProject Atlas is owned by the VisionCore Team." in document.text
    assert "<p>" not in document.text


def test_html_scripts_are_removed(processor, tmp_path: Path) -> None:
    path = tmp_path / "x.html"
    path.write_text(
        "<html><head><title>T</title><script>alert('x')</script></head>"
        "<body><p>Hello.</p></body></html>",
        encoding="utf-8",
    )
    document = processor.parse(path, 1, "x.html")
    assert document.text == "Hello."
    assert document.metadata["title"] == "T"


def test_text_is_normalised(processor, tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("Acme    AI   developed\tVisionCore.\r\n\r\n\r\nSecond line.", encoding="utf-8")
    document = processor.parse(path, 1, "x.txt")
    assert document.text == "Acme AI developed VisionCore.\nSecond line."


@pytest.mark.parametrize(
    ("name", "content"),
    [("broken.pdf", b"%PDF-1.7 broken"), ("broken.docx", b"PK\x03\x04junk"), ("blank.txt", b"  \n ")],
)
def test_malformed_documents_raise_a_clear_error(processor, tmp_path: Path, name, content) -> None:
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(DocumentParsingError):
        processor.parse(path, 1, name)


def test_chunk_offsets_point_into_the_text(processor, sample_files) -> None:
    for number, path in enumerate(sample_files, start=1):
        document = processor.parse(path, number, path.name)
        chunks = processor.chunk(document)
        assert chunks
        for chunk in chunks:
            assert document.text[chunk.start : chunk.end] == chunk.text
        assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_sentence_offsets_and_abbreviations() -> None:
    text = "Dr. Sarah Chen leads Atlas. Northwind Inc. uses it.\nA heading"
    sentences = split_sentences(text)
    assert [s for _, _, s in sentences] == [
        "Dr. Sarah Chen leads Atlas.",
        "Northwind Inc. uses it.",
        "A heading",
    ]
    assert all(text[start:end] == sentence for start, end, sentence in sentences)
