"""Document and ingestion-run bookkeeping in SQLite (application-owned)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from backend.config import Settings
from backend.ingestion.validation import validate_upload
from database.models import Document, DocumentStatus, IngestionRun, Query, QuerySource, utcnow
from database.session import Database


@dataclass(frozen=True)
class Registration:
    document: Document
    created: bool  # False when identical content was already registered


class DocumentRegistry:
    def __init__(self, database: Database, settings: Settings) -> None:
        self._db = database
        self._settings = settings

    # -- documents -----------------------------------------------------------
    def register(self, filename: str, content: bytes) -> Registration:
        """Validate and store an upload; identical content is never stored twice."""
        upload = validate_upload(filename, content, self._settings.max_upload_bytes)
        with self._db.session() as session:
            existing = session.scalar(select(Document).where(Document.file_hash == upload.sha256))
            if existing is not None:
                return Registration(existing, created=False)

            self._settings.uploads_dir.mkdir(parents=True, exist_ok=True)
            path = self._settings.uploads_dir / upload.stored_filename
            path.write_bytes(content)
            document = Document(
                original_filename=upload.original_filename,
                stored_filename=upload.stored_filename,
                file_type=upload.extension.lstrip("."),
                file_path=str(path),
                file_hash=upload.sha256,
                file_size=upload.size,
            )
            session.add(document)
            session.flush()
            return Registration(document, created=True)

    def get(self, document_id: int) -> Document:
        with self._db.session() as session:
            document = session.get(Document, document_id, options=[selectinload(Document.runs)])
            if document is None:
                raise KeyError(f"No document with id {document_id}")
            return document

    def list_documents(self) -> list[Document]:
        with self._db.session() as session:
            statement = select(Document).options(selectinload(Document.runs)).order_by(Document.id)
            return list(session.scalars(statement))

    def delete(self, document_id: int) -> None:
        with self._db.session() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise KeyError(f"No document with id {document_id}")
            path = Path(document.file_path)
            session.delete(document)
        path.unlink(missing_ok=True)

    # -- ingestion runs ------------------------------------------------------
    def start_run(self, document_id: int, method: str) -> int:
        with self._db.session() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise KeyError(f"No document with id {document_id}")
            document.processing_status = DocumentStatus.PROCESSING
            document.error_message = None
            run = IngestionRun(
                document_id=document_id,
                status=DocumentStatus.PROCESSING,
                extraction_method=method,
            )
            session.add(run)
            session.flush()
            return run.id

    def finish_run(
        self,
        run_id: int,
        *,
        duration: float,
        entities: int = 0,
        relationships: int = 0,
        error: str | None = None,
    ) -> None:
        status = DocumentStatus.FAILED if error else DocumentStatus.PROCESSED
        with self._db.session() as session:
            run = session.get(IngestionRun, run_id)
            if run is None:
                raise KeyError(f"No ingestion run with id {run_id}")
            run.status = status
            run.entities_count = entities
            run.relationships_count = relationships
            run.processing_duration = duration
            run.completed_at = utcnow()
            run.error_message = error
            run.document.processing_status = status
            run.document.error_message = error
            if not error:
                run.document.processed_at = utcnow()

    def recent_runs(self, limit: int = 20) -> list[IngestionRun]:
        with self._db.session() as session:
            statement = (
                select(IngestionRun)
                .options(selectinload(IngestionRun.document))
                .order_by(IngestionRun.id.desc())
                .limit(limit)
            )
            return list(session.scalars(statement))

    def counts(self) -> dict[str, int]:
        with self._db.session() as session:
            return {
                "documents": session.scalar(select(func.count(Document.id))) or 0,
                "ingestion_runs": session.scalar(select(func.count(IngestionRun.id))) or 0,
                "queries": session.scalar(select(func.count(Query.id))) or 0,
            }

    # -- query log -----------------------------------------------------------
    def log_query(
        self,
        question: str,
        mode: str,
        response: str,
        execution_time: float,
        sources: list[dict],
    ) -> int:
        """Persist a query and the evidence that supported it.

        Each source dict needs ``document_id``, ``source_type``,
        ``evidence_reference`` and optionally ``relevance_score``.
        """
        with self._db.session() as session:
            known = set(session.scalars(select(Document.id)))
            query = Query(
                question=question,
                retrieval_mode=mode,
                response=response,
                execution_time=execution_time,
            )
            query.sources = [
                QuerySource(
                    document_id=source["document_id"],
                    relevance_score=source.get("relevance_score"),
                    source_type=source["source_type"],
                    evidence_reference=source["evidence_reference"],
                )
                for source in sources
                if source.get("document_id") in known
            ]
            session.add(query)
            session.flush()
            return query.id
