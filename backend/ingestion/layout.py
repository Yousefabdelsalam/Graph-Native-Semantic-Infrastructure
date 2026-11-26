"""Layout-aware reading of PDF pages, on top of ``pdfplumber``.

Semantica 0.7.0's ``PDFParser`` returns ``page.extract_text()``: side-by-side
columns are interleaved line by line, table cells are scrambled and running
headers and footers become body text.  Evidence quoted from such text mixes
unrelated statements.  This module returns the page as ordered *blocks* instead:

* columns are read one after the other,
* ruled tables become one ``column: value`` block per row,
* headings are marked, with the section each piece of text belongs to,
* lines repeated in the top or bottom margin of most pages are marked as
  furniture and left out of the document text.

Page numbers are preserved.  The rules are heuristics for digitally created
PDFs; scanned pages still need OCR.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pdfplumber

MARGIN = 0.10  # share of the page height treated as header / footer area
_LINE_TOLERANCE = 3.0  # points between the tops of words on one line
_MIN_GUTTER = 14.0  # horizontal gap that separates two columns
_INDENT = 24.0  # largest indentation within one paragraph
_BULLET = re.compile(r"^(?:[•▪◦‣●■\-–—*]\s+|\(?\d{1,2}[.)]\s+|\(?[a-z][.)]\s+)")
_BOLD = re.compile(r"bold|black|heavy|semibold|demi", re.IGNORECASE)


@dataclass
class RawBlock:
    page: int
    kind: str  # "title" | "heading" | "text" | "table" | "furniture"
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float = 0.0
    level: int = 0  # headings: 1 = section, 2 = subsection
    path: list[int] = field(default_factory=list)  # indices of the enclosing headings
    columnar: bool = False  # sits beside another block


@dataclass
class _Segment:
    """Words of one line that belong to the same column."""

    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    bold: bool
    split: bool = False  # the line held more than one segment


@dataclass
class _Paragraph:
    lines: list[_Segment]

    @property
    def x0(self) -> float:
        return min(line.x0 for line in self.lines)

    @property
    def x1(self) -> float:
        return max(line.x1 for line in self.lines)


def read_pdf(path: Path) -> list[RawBlock]:
    """All blocks of a PDF in reading order, with headings and furniture marked."""
    blocks: list[RawBlock] = []
    margins: list[tuple[RawBlock, float]] = []  # block, page height
    with pdfplumber.open(str(path)) as pdf:
        pages = len(pdf.pages)
        for number, page in enumerate(pdf.pages, start=1):
            height = float(page.height)
            found = _page_blocks(page, number)
            blocks += found
            margins += [
                (b, height)
                for b in found
                if b.kind != "table"
                and (b.bottom <= height * MARGIN or b.top >= height * (1 - MARGIN))
            ]
    _mark_furniture(margins, pages)
    _mark_headings(blocks)
    ordered = _reading_order(blocks)
    _assign_sections(ordered)
    return ordered


# ------------------------------------------------------------------- one page
def _page_blocks(page, number: int) -> list[RawBlock]:
    tables = []
    for table in page.find_tables():
        rows = [[_clean(cell) for cell in row] for row in table.extract()]
        # A one-row or one-column "table" is a framed box, not tabular data.
        if len(rows) >= 2 and max(len(r) for r in rows) >= 2 and sum(map(bool, rows[0])) >= 2:
            tables.append((table.bbox, rows))

    def in_table(word: dict) -> bool:
        x, y = (word["x0"] + word["x1"]) / 2, (word["top"] + word["bottom"]) / 2
        return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b, _ in tables)

    words = [
        w for w in page.extract_words(extra_attrs=["size", "fontname"]) if not in_table(w)
    ]
    blocks = [
        RawBlock(
            page=number,
            kind="text",
            text=_join_lines([line.text for line in paragraph.lines]),
            x0=paragraph.x0,
            x1=paragraph.x1,
            top=paragraph.lines[0].top,
            bottom=paragraph.lines[-1].bottom,
            size=paragraph.lines[0].size,
            level=1 if paragraph.lines[0].bold else 0,  # provisional: "is bold"
        )
        for paragraph in _paragraphs(_segments(words))
    ]
    for bbox, rows in tables:
        blocks += _table_rows(number, bbox, rows)
    return sorted(blocks, key=lambda b: (b.top, b.x0))


def _segments(words: list[dict]) -> list[_Segment]:
    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and word["top"] - lines[-1][0]["top"] <= _LINE_TOLERANCE:
            lines[-1].append(word)
        else:
            lines.append([word])
    segments: list[_Segment] = []
    for line in lines:
        line.sort(key=lambda w: w["x0"])
        groups = [[line[0]]]
        for word in line[1:]:
            previous = groups[-1][-1]
            gutter = max(_MIN_GUTTER, 1.4 * previous["size"])
            if word["x0"] - previous["x1"] > gutter:
                groups.append([word])
            else:
                groups[-1].append(word)
        for group in groups:
            bold = sum(bool(_BOLD.search(w.get("fontname", ""))) for w in group)
            segments.append(
                _Segment(
                    text=" ".join(w["text"] for w in group),
                    x0=group[0]["x0"],
                    x1=group[-1]["x1"],
                    top=min(w["top"] for w in group),
                    bottom=max(w["bottom"] for w in group),
                    size=round(max(w["size"] for w in group), 1),
                    bold=bold * 2 > len(group),
                    split=len(groups) > 1,
                )
            )
    return segments


def _paragraphs(segments: Iterable[_Segment]) -> list[_Paragraph]:
    """Stack segments of the same column and style into paragraphs."""
    paragraphs: list[_Paragraph] = []
    for segment in segments:
        for paragraph in reversed(paragraphs):
            if _continues(paragraph, segment):
                paragraph.lines.append(segment)
                break
        else:
            paragraphs.append(_Paragraph([segment]))
    return paragraphs


def _continues(paragraph: _Paragraph, segment: _Segment) -> bool:
    last = paragraph.lines[-1]
    overlap = min(paragraph.x1, segment.x1) - max(paragraph.x0, segment.x0)
    if not (
        0 < segment.top - last.top <= last.size * 1.9
        and abs(segment.x0 - paragraph.x0) <= _INDENT
        and overlap > 0
        and segment.bold == last.bold
        and abs(segment.size - last.size) < 0.6
        and not _BULLET.match(segment.text)
    ):
        return False
    if segment.bold or segment.text[:1].islower():
        return True
    # A new line that starts in upper case continues the paragraph only if the
    # previous line was full; short lines are separate list entries.
    width = max(paragraph.x1, segment.x1) - paragraph.x0
    return last.x1 - paragraph.x0 >= 0.7 * width


def _table_rows(page: int, bbox: tuple, rows: list[list[str]]) -> list[RawBlock]:
    header, body = rows[0], rows[1:]
    step = (bbox[3] - bbox[1]) / len(rows)
    blocks = []
    for index, row in enumerate(body, start=1):
        cells = [
            f"{name}: {value.rstrip('.')}" if name else value.rstrip(".")
            for name, value in zip(header, row)
            if value
        ]
        if cells:
            blocks.append(
                RawBlock(
                    page=page,
                    kind="table",
                    text="; ".join(cells) + ".",
                    x0=bbox[0],
                    x1=bbox[2],
                    top=bbox[1] + step * index,
                    bottom=bbox[1] + step * (index + 1),
                )
            )
    return blocks


def _clean(cell: object) -> str:
    return _join_lines(str(cell or "").splitlines())


def _join_lines(lines: list[str]) -> str:
    text = ""
    for line in (" ".join(part.split()) for part in lines):
        if not line:
            continue
        # "goal-" + "directed" is one hyphenated word.
        glue = "" if not text or (text.endswith("-") and line[:1].islower()) else " "
        text += glue + line
    return text


# ------------------------------------------------------------- whole document
def _mark_furniture(margins: list[tuple[RawBlock, float]], pages: int) -> None:
    """Lines repeated in the margins of at least half the pages (min. two)."""
    if pages < 2:
        return

    def signature(block: RawBlock, height: float) -> tuple[str, bool]:
        return re.sub(r"\d+", "#", block.text.lower()), block.top < height / 2

    seen: dict[tuple[str, bool], set[int]] = {}
    for block, height in margins:
        seen.setdefault(signature(block, height), set()).add(block.page)
    needed = max(2, math.ceil(pages / 2))
    for block, height in margins:
        if len(seen[signature(block, height)]) >= needed:
            block.kind = "furniture"


def _mark_headings(blocks: list[RawBlock]) -> None:
    weights: Counter[float] = Counter()
    for block in blocks:
        if block.kind == "text" and not block.level:
            weights[block.size] += len(block.text)
    body_size = weights.most_common(1)[0][0] if weights else 0.0
    all_bold = not weights
    for block in blocks:
        bold, block.level = bool(block.level), 0
        if block.kind != "text" or all_bold:
            continue
        larger = bool(body_size) and block.size >= body_size + 1.5
        if (bold or larger) and len(block.text) <= 160 and not block.text.endswith("."):
            block.kind = "heading"

    # The document title: the single largest heading of the first page, and any
    # front-matter headings between it and the section heading that precedes the text.
    headings = [b for b in blocks if b.kind == "heading"]
    if not headings:
        return
    largest = max(h.size for h in headings)
    top = [h for h in headings if h.size == largest]
    if len(top) != 1 or top[0].page != 1:
        return
    top[0].kind = "title"
    first_page = [b for b in blocks if b.page == 1 and b.kind != "furniture"]
    start = next(i for i, b in enumerate(first_page) if b is top[0]) + 1
    run = []
    for block in first_page[start:]:
        if block.kind != "heading":
            break
        run.append(block)
    for block in run[:-1]:
        block.kind = "title"


def _reading_order(blocks: list[RawBlock]) -> list[RawBlock]:
    """Within a page, finish one column before starting the next."""
    ordered: list[RawBlock] = []
    for page in sorted({b.page for b in blocks}):
        items = sorted((b for b in blocks if b.page == page), key=lambda b: (b.top, b.x0))
        content = [b for b in items if b.kind != "furniture"]
        for block in content:
            block.columnar = any(
                other is not block
                and other.kind != "table"
                and block.kind != "table"
                and min(block.bottom, other.bottom) - max(block.top, other.top) > 0
                and min(block.x1, other.x1) - max(block.x0, other.x0) <= 0
                for other in content
            )
        run: list[RawBlock] = []

        def flush() -> None:
            columns: list[list[RawBlock]] = []
            for block in sorted(run, key=lambda b: b.x0):
                for column in columns:
                    if abs(block.x0 - column[0].x0) <= _INDENT:
                        column.append(block)
                        break
                else:
                    columns.append([block])
            for column in columns:
                ordered.extend(sorted(column, key=lambda b: b.top))
            run.clear()

        for block in content:
            if block.columnar:
                run.append(block)
            else:
                flush()
                ordered.append(block)
        flush()
        ordered += [b for b in items if b.kind == "furniture"]
    return ordered


def _assign_sections(blocks: list[RawBlock]) -> None:
    """Record which section (and subsection) heading each block belongs to."""
    headings = [b for b in blocks if b.kind == "heading" and not b.columnar]
    section_size = max((h.size for h in headings), default=0.0)
    section: int | None = None
    sub: int | None = None
    sub_x: float | None = None  # left edge of the first text under the subsection
    for index, block in enumerate(blocks):
        if block.kind in ("furniture", "title"):
            continue
        if block.kind == "heading":
            if not block.columnar and block.size >= section_size - 0.5:
                block.level, section, sub = 1, index, None
            else:
                block.level, sub, sub_x = 2, index, None
                block.path = [section] if section is not None else []
            continue
        if sub is not None:
            heading = blocks[sub]
            overlaps = min(block.x1, heading.x1) - max(block.x0, heading.x0) > 0
            if sub_x is None and overlaps and block.page == heading.page:
                sub_x = block.x0
            elif sub_x is None or abs(block.x0 - sub_x) > 3:
                sub = None  # back to the enclosing section
        block.path = [i for i in (section, sub) if i is not None]
