"""Ingestion, parsing, normalisation and chunking.

Semantica does the work: ``FileIngestor`` (file intake), ``PDFParser`` /
``DOCXParser`` / ``HTMLParser`` / ``DocumentParser`` (text extraction),
``TextNormalizer`` and ``TextSplitter``.  The application adds one thing the
library does not return: character offsets of pages/sections inside the
normalised text, so every fact can point back to a location.  PDF text is read
by :mod:`backend.ingestion.layout` (columns, tables, headers and footers);
Semantica's ``PDFParser`` still supplies the document metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from semantica.ingest import FileIngestor
from semantica.normalize import TextNormalizer
from semantica.parse import DocumentParser, DOCXParser, HTMLParser, PDFParser
from semantica.split import TextSplitter

from backend.ingestion.layout import RawBlock, read_pdf
from backend.schemas import Block, Chunk, ParsedDocument, Segment

logger = logging.getLogger(__name__)

_SEPARATOR = "\n\n"
_BLOCK_TAGS = ["p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6", "title"]


class DocumentParsingError(RuntimeError):
    """The file could not be turned into text."""


class DocumentProcessor:
    def __init__(self, chunk_size: int = 420, chunk_overlap: int = 80) -> None:
        self._ingestor = FileIngestor()
        self._parser = DocumentParser()
        self._pdf = PDFParser()
        self._docx = DOCXParser()
        self._html = HTMLParser()
        self._normalizer = TextNormalizer()
        self._splitter = TextSplitter(
            method="recursive", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    # -- parsing -------------------------------------------------------------
    def parse(self, path: Path, document_id: int, filename: str) -> ParsedDocument:
        """Ingest and parse one stored file into normalised, located text."""
        try:
            file_object = self._ingestor.ingest_file(path, read_content=False)
            extension = path.suffix.lower()
            if extension == ".pdf":
                return self._parse_pdf(path, document_id, filename, file_object)
            elif extension == ".docx":
                raw_segments, metadata, parser = self._parse_docx(path)
            elif extension in (".html", ".htm"):
                raw_segments, metadata, parser = self._parse_html(path)
            else:
                raw_segments, metadata, parser = self._parse_text(path)
        except Exception as exc:  # Semantica raises several exception types
            raise DocumentParsingError(f"Could not parse '{filename}': {exc}") from exc

        text_parts: list[str] = []
        segments: list[Segment] = []
        offset = 0
        for label, raw in raw_segments:
            cleaned = self.normalize(raw)
            if not cleaned:
                continue
            segments.append(Segment(label=label, start=offset, end=offset + len(cleaned)))
            text_parts.append(cleaned)
            offset += len(cleaned) + len(_SEPARATOR)

        if not segments:
            raise DocumentParsingError(
                f"'{filename}' contains no extractable text (scanned PDFs need OCR)."
            )
        metadata["size_bytes"] = str(getattr(file_object, "size", path.stat().st_size))
        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            file_type=extension.lstrip("."),
            text=_SEPARATOR.join(text_parts),
            segments=segments,
            metadata=metadata,
            parser=parser,
        )

    def normalize(self, text: str) -> str:
        """Semantica normalisation applied line by line so line breaks survive."""
        lines = (self._normalizer.normalize_text(line).strip() for line in text.splitlines())
        return "\n".join(line for line in lines if line)

    def _parse_pdf(
        self, path: Path, document_id: int, filename: str, file_object: Any
    ) -> ParsedDocument:
        """One segment per page, built from layout blocks without headers / footers."""
        metadata = _clean_metadata(
            self._pdf.parse(path, extract_text=False, extract_tables=False).get("metadata")
        )
        text, segments, blocks = self._assemble(read_pdf(path))
        if not segments:
            raise DocumentParsingError(
                f"'{filename}' contains no extractable text (scanned PDFs need OCR)."
            )
        metadata["size_bytes"] = str(getattr(file_object, "size", path.stat().st_size))
        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            file_type="pdf",
            text=text,
            segments=segments,
            metadata=metadata,
            parser="pdfplumber layout reader (app); metadata: semantica.parse.PDFParser",
            blocks=blocks,
        )

    def _assemble(self, raw: list[RawBlock]) -> tuple[str, list[Segment], list[Block]]:
        """Normalise each block and place it in the document text.

        Blocks are separated by a blank line, so a sentence can never run from
        one block into the next.
        """
        text = ""
        blocks: list[Block] = []
        segments: dict[str, Segment] = {}
        position: dict[int, int] = {}  # index in ``raw`` -> index in ``blocks``
        for index, item in enumerate(raw):
            cleaned = " ".join(self.normalize(item.text).split())
            if not cleaned:
                continue
            label = f"page {item.page}"
            start = end = None
            if item.kind != "furniture":
                start = len(text) + (len(_SEPARATOR) if text else 0)
                end = start + len(cleaned)
                text += (_SEPARATOR if text else "") + cleaned
                segment = segments.setdefault(label, Segment(label=label, start=start, end=end))
                segment.end = end
            position[index] = len(blocks)
            blocks.append(
                Block(
                    segment=label,
                    kind=item.kind,
                    text=cleaned,
                    start=start,
                    end=end,
                    level=item.level,
                    path=[position[i] for i in item.path if i in position],
                )
            )
        return text, list(segments.values()), blocks

    def _parse_docx(self, path: Path) -> tuple[list[tuple[str, str]], dict[str, str], str]:
        result = self._docx.parse(path)
        paragraphs = [p for p in result.get("paragraphs", []) if str(p).strip()]
        headings = {
            str(section.get("heading") or section.get("title") or "").strip()
            for section in result.get("sections", [])
        } - {""}
        segments: list[tuple[str, str]] = []
        current, buffer = "body", []
        for paragraph in (str(p) for p in paragraphs):
            if paragraph.strip() in headings:
                if buffer:
                    segments.append((current, "\n".join(buffer)))
                current, buffer = f"section '{paragraph.strip()}'", []
            buffer.append(paragraph)
        if buffer:
            segments.append((current, "\n".join(buffer)))
        if not segments and result.get("full_text"):
            segments = [("body", result["full_text"])]
        return segments, _clean_metadata(result.get("metadata")), "semantica.parse.DOCXParser"

    def _parse_html(self, path: Path) -> tuple[list[tuple[str, str]], dict[str, str], str]:
        """Parse HTML while keeping block boundaries.

        Semantica 0.7.0's default ``clean_text=True`` joins every line with a
        space, which glues headings onto the following sentence.  We therefore
        ask for the raw text (``clean_text=False``) and remove non-content
        elements first; ``<br>``/block tags get explicit line breaks.
        """
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else ""
        for element in soup(["script", "style", "noscript", "title"]):
            element.decompose()
        for element in soup(_BLOCK_TAGS):
            element.append("\n")
        result = self._html.parse(
            str(soup), clean_text=False, extract_links=False, extract_images=False
        )
        metadata = {"title": title} if title else {}
        return [("body", result.text)], metadata, "semantica.parse.HTMLParser"

    def _parse_text(self, path: Path) -> tuple[list[tuple[str, str]], dict[str, str], str]:
        result = self._parser.parse_document(path, file_type="text")
        return [("body", result["full_text"])], {}, "semantica.parse.DocumentParser"

    # -- chunking ------------------------------------------------------------
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Split each located segment with Semantica's recursive splitter."""
        chunks: list[Chunk] = []
        for segment in document.segments:
            body = document.text[segment.start : segment.end]
            for piece in self._splitter.split(body):
                text = piece.text.strip()
                if not text:
                    continue
                start = segment.start + piece.start_index + (len(piece.text) - len(piece.text.lstrip()))
                chunks.append(
                    Chunk(
                        chunk_id=f"doc{document.document_id}-c{len(chunks)}",
                        document_id=document.document_id,
                        filename=document.filename,
                        text=text,
                        start=start,
                        end=start + len(text),
                        location=f"{segment.label}, chars {start}-{start + len(text)}",
                    )
                )
        return chunks


def _clean_metadata(metadata: Any) -> dict[str, str]:
    if not isinstance(metadata, dict):
        return {}
    return {str(k): str(v) for k, v in metadata.items() if v not in (None, "", [], {})}


def location_of(document: ParsedDocument, start: int, end: int) -> str:
    """Human-readable location of a character span."""
    label = next(
        (s.label for s in document.segments if s.start <= start < s.end),
        "body",
    )
    return f"{label}, chars {start}-{end}"
