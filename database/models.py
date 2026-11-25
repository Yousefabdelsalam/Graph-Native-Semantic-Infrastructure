"""SQLite application metadata.

Only *application* bookkeeping lives here.  The knowledge graph itself is owned
by Semantica + Neo4j, and fact provenance by Semantica's ``ProvenanceManager``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class DocumentStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    # SHA-256 of the file bytes; the uniqueness constraint is what makes
    # re-uploading unchanged content a no-op.
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    processing_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DocumentStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)

    runs: Mapped[list["IngestionRun"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="IngestionRun.id"
    )
    query_sources: Mapped[list["QuerySource"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_documents_status", "processing_status"),)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(32), nullable=False)
    entities_count: Mapped[int] = mapped_column(Integer, default=0)
    relationships_count: Mapped[int] = mapped_column(Integer, default=0)
    processing_duration: Mapped[float | None] = mapped_column(Float)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="runs")

    __table_args__ = (
        Index("ix_ingestion_runs_document", "document_id"),
        Index("ix_ingestion_runs_started", "started_at"),
    )


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    response: Mapped[str | None] = mapped_column(Text)
    execution_time: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    sources: Mapped[list["QuerySource"]] = relationship(
        back_populates="query", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_queries_mode", "retrieval_mode"),
        Index("ix_queries_created", "created_at"),
    )


class QuerySource(Base):
    __tablename__ = "query_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query_id: Mapped[int] = mapped_column(
        ForeignKey("queries.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    relevance_score: Mapped[float | None] = mapped_column(Float)
    source_type: Mapped[str] = mapped_column(String(16), nullable=False)  # chunk | fact
    evidence_reference: Mapped[str] = mapped_column(Text, nullable=False)

    query: Mapped[Query] = relationship(back_populates="sources")
    document: Mapped[Document] = relationship(back_populates="query_sources")

    __table_args__ = (
        Index("ix_query_sources_query", "query_id"),
        Index("ix_query_sources_document", "document_id"),
    )
