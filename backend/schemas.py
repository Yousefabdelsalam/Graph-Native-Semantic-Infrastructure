"""Pydantic contracts passed between pipeline stages.

These are application-owned.  They are intentionally thin wrappers around the
dictionaries Semantica consumes and produces, so each stage can be validated and
the per-document extraction output can be persisted as a JSON artifact.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """A located span of the normalised document text (a page or a section)."""

    label: str
    start: int
    end: int


class Block(BaseModel):
    """A heading, paragraph, table row or running header / footer.

    ``start`` / ``end`` are offsets into ``ParsedDocument.text``; they are empty
    for furniture, which is not part of the text.
    """

    segment: str  # label of the page or section holding the block
    kind: str  # "title" | "heading" | "text" | "table" | "furniture"
    text: str
    start: int | None = None
    end: int | None = None
    level: int = 0  # headings: 1 = section, 2 = subsection
    path: list[int] = Field(default_factory=list)  # indices of enclosing headings


class ParsedDocument(BaseModel):
    document_id: int
    filename: str
    file_type: str
    text: str
    segments: list[Segment]
    metadata: dict[str, str] = Field(default_factory=dict)
    parser: str
    blocks: list[Block] = Field(default_factory=list)  # empty for non-PDF files


class Chunk(BaseModel):
    chunk_id: str
    document_id: int
    filename: str
    text: str
    start: int
    end: int
    location: str


class Mention(BaseModel):
    """One entity occurrence found by a Semantica extractor."""

    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    method: str
    location: str


class Fact(BaseModel):
    """One relationship or attribute statement, tied to the sentence stating it."""

    fact_id: str
    kind: str  # "relation" | "attribute"
    subject: str  # surface text; resolved to a canonical entity during graph build
    subject_type: str
    predicate: str
    object: str
    object_type: str  # "Literal" for attribute values
    quote: str
    location: str
    start: int
    end: int
    confidence: float
    method: str
    extracted_at: str
    valid_from: str | None = None
    valid_until: str | None = None
    # Set by evidence validation on a working copy; never part of the raw result.
    support: str | None = None  # "supported" | "inferred"
    support_reason: str | None = None


class DocumentExtraction(BaseModel):
    """Everything extracted from one document; stored as ``<document_id>.json``."""

    document_id: int
    filename: str
    method: str
    extracted_at: str
    mentions: list[Mention]
    facts: list[Fact]
    events: list[dict] = Field(default_factory=list)
    triplets: list[list[str]] = Field(default_factory=list)
    discarded: dict[str, int] = Field(default_factory=dict)


class Evidence(BaseModel):
    """A single retrievable piece of support shown to the user."""

    kind: str  # "chunk" | "fact" | "conflict" | "context"
    text: str
    score: float | None = None
    document_ids: list[int] = Field(default_factory=list)
    filenames: list[str] = Field(default_factory=list)
    location: str | None = None
    reference: str
    support: str | None = None  # graph facts: "supported" | "inferred"


class RetrievalResult(BaseModel):
    question: str
    mode: str
    latency_ms: float
    entities: list[dict] = Field(default_factory=list)
    relationships: list[dict] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    sufficient: bool = True
    note: str = ""

    @property
    def source_files(self) -> list[str]:
        return sorted({name for item in self.evidence for name in item.filenames})
