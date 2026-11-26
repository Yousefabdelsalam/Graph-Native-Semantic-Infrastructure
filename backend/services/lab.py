"""The pipeline orchestrator used by the UI, the scripts and the tests.

upload → parse → normalise → extract → validate evidence → resolve / detect
conflicts → build graph → Neo4j sync → vector index → provenance

Neo4j holds the graph.  Per-document extraction results are kept as JSON
artifacts so the graph can be re-derived whenever any document changes.  Those
raw results are never rewritten by a rebuild: the cleaned layout text and the
evidence verdicts are saved beside them (``layout/`` and ``validation/``).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from backend.config import Settings, load_settings
from backend.extraction.extractor import ExtractionPipeline
from backend.extraction.validation import ValidationReport, validate
from backend.graph.builder import AssembledGraph, assemble_graph
from backend.graph.conflicts import detect_conflicts
from backend.graph.repository import GraphRepository
from backend.ingestion.parser import DocumentProcessor
from backend.provenance.tracker import ProvenanceTracker
from backend.retrieval.answer import generate_answer
from backend.retrieval.engine import RetrievalEngine
from backend.retrieval.vector import VectorIndex
from backend.schemas import DocumentExtraction, ParsedDocument, RetrievalResult
from backend.services.registry import DocumentRegistry
from database.models import Document, DocumentStatus
from database.session import Database

logger = logging.getLogger(__name__)


@dataclass
class IngestOutcome:
    document_id: int
    filename: str
    status: str  # "processed" | "skipped" | "failed"
    message: str = ""
    entities: int = 0
    relationships: int = 0


class KnowledgeLab:
    def __init__(
        self, settings: Settings | None = None, repository: GraphRepository | None = None
    ) -> None:
        self.settings = settings or load_settings()
        self.settings.ensure_directories()
        self.database = Database(self.settings.metadata_db)
        self.registry = DocumentRegistry(self.database, self.settings)
        self.processor = DocumentProcessor()
        self.provenance = ProvenanceTracker(self.settings.provenance_db)
        self.repository = repository or GraphRepository(self.settings)
        self._vector: VectorIndex | None = None
        self._lock = threading.RLock()

    @property
    def vector(self) -> VectorIndex:
        if self._vector is None:
            self._vector = VectorIndex(self.settings)
        return self._vector

    def close(self) -> None:
        self.repository.close()
        self.database.dispose()

    # ---------------------------------------------------------------- ingestion
    def ingest(self, filename: str, content: bytes, method: str = "regex") -> IngestOutcome:
        """Register and process an upload; unchanged, processed content is skipped."""
        with self._lock:
            registration = self.registry.register(filename, content)
            document = registration.document
            if (
                not registration.created
                and document.processing_status == DocumentStatus.PROCESSED
            ):
                return IngestOutcome(
                    document.id,
                    document.original_filename,
                    "skipped",
                    f"Identical content was already processed as '{document.original_filename}'.",
                )
            return self.process(document.id, method)

    def process(self, document_id: int, method: str = "regex") -> IngestOutcome:
        """(Re)process a registered document and update every derived store."""
        with self._lock:
            document = self.registry.get(document_id)
            run_id = self.registry.start_run(document_id, method)
            started = time.perf_counter()
            try:
                parsed, extraction, assembled = self._run_pipeline(document, method)
            except Exception as exc:
                logger.exception("Processing document %s failed", document_id)
                message = f"{type(exc).__name__}: {exc}"
                self.registry.finish_run(
                    run_id, duration=time.perf_counter() - started, error=message
                )
                return IngestOutcome(document_id, document.original_filename, "failed", message)
            entities = len(
                {
                    assembled.resolution.resolve(m.entity_type, m.text)
                    for m in extraction.mentions
                }
                - {None}
            )
            relationships = sum(1 for f in extraction.facts if f.kind == "relation")
            self.registry.finish_run(
                run_id,
                duration=time.perf_counter() - started,
                entities=entities,
                relationships=relationships,
            )
            return IngestOutcome(
                document_id,
                parsed.filename,
                "processed",
                entities=entities,
                relationships=relationships,
            )

    def _run_pipeline(
        self, document: Document, method: str
    ) -> tuple[ParsedDocument, DocumentExtraction, AssembledGraph]:
        parsed = self.processor.parse(
            Path(document.file_path), document.id, document.original_filename
        )
        extraction = ExtractionPipeline(self.settings, method).extract(parsed)
        chunks = self.processor.chunk(parsed)
        previous = self.extraction(document.id)
        vector = self.vector  # fail early if the embedding model is unavailable

        others = [e for e in self._extractions() if e.document_id != document.id]
        # Neo4j first: most likely to fail
        assembled, reports = self._rebuild([*others, extraction], {document.id: parsed})

        vector.replace_document(document.id, chunks)
        self.provenance.record_document(parsed, document.file_hash)
        self.provenance.record_extraction(extraction)
        self._record_validation([*others, extraction], reports)
        if previous:
            gone = {f.fact_id for f in previous.facts} - {f.fact_id for f in extraction.facts}
            self.provenance.invalidate(sorted(gone), "No longer extracted after reprocessing")
        self._write_json(self._extraction_path(document.id), extraction.model_dump())
        self._write_json(self.settings.exports_dir / f"parsed_{document.id}.json", parsed.model_dump())
        return parsed, extraction, assembled

    def delete_document(self, document_id: int) -> None:
        with self._lock:
            self.registry.get(document_id)
            extraction = self.extraction(document_id)
            remaining = [e for e in self._extractions() if e.document_id != document_id]
            _, reports = self._rebuild(remaining)
            self._record_validation(remaining, reports)
            self.vector.remove_document(document_id)
            if extraction:
                self.provenance.invalidate(
                    [f.fact_id for f in extraction.facts], "Source document deleted"
                )
            self._extraction_path(document_id).unlink(missing_ok=True)
            (self.settings.exports_dir / f"parsed_{document_id}.json").unlink(missing_ok=True)
            self._layout_path(document_id).unlink(missing_ok=True)
            self._validation_path(document_id).unlink(missing_ok=True)
            self.registry.delete(document_id)

    def load_sample_data(self, method: str = "regex") -> list[IngestOutcome]:
        files = sorted(
            p for p in self.settings.sample_dir.iterdir() if p.suffix.lower() != ".json"
        )
        return [self.ingest(path.name, path.read_bytes(), method) for path in files]

    def reset(self) -> None:
        """Remove all documents and derived data.  Provenance history is kept."""
        with self._lock:
            for document in self.registry.list_documents():
                self.delete_document(document.id)

    # ------------------------------------------------------------------- graph
    def rebuild_graph(self) -> AssembledGraph:
        """Re-derive the graph from stored extraction results; nothing is re-extracted."""
        with self._lock:
            extractions = self._extractions()
            assembled, reports = self._rebuild(extractions)
            self._record_validation(extractions, reports)
            return assembled

    def refresh_search_index(self, document_id: int) -> int:
        """Re-index one document's search chunks from its cleaned layout text."""
        with self._lock:
            chunks = self.processor.chunk(self.layout(document_id))
            self.vector.replace_document(document_id, chunks)
            return len(chunks)

    def layout(self, document_id: int) -> ParsedDocument:
        """The cleaned, block-structured text of a stored document."""
        path = self._layout_path(document_id)
        if path.exists():
            return ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        document = self.registry.get(document_id)
        return self.processor.parse(
            Path(document.file_path), document.id, document.original_filename
        )

    def _rebuild(
        self,
        extractions: list[DocumentExtraction],
        fresh: dict[int, ParsedDocument] | None = None,
    ) -> tuple[AssembledGraph, dict[int, ValidationReport]]:
        """Validate evidence, assemble the graph and synchronise Neo4j.

        A document without any extracted fact has nothing to validate: its file is
        not read again.
        """
        layouts: dict[int, ParsedDocument] = dict(fresh or {})
        reports: dict[int, ValidationReport] = {}
        working = []
        for extraction in sorted(extractions, key=lambda e: e.document_id):
            if extraction.facts:
                document_id = extraction.document_id
                if document_id not in layouts:
                    layouts[document_id] = self.layout(document_id)
                reports[document_id] = validate(extraction, layouts[document_id])
                extraction = reports[document_id].extraction
            working.append(extraction)

        assembled = assemble_graph(working)
        conflicts, successions = detect_conflicts(assembled.observations)
        sync = self.repository.sync(assembled.as_dict())
        self._write_json(
            self.settings.exports_dir / "analysis.json",
            {
                "conflicts": conflicts,
                "successions": successions,
                "observations": assembled.observations,
                "merges": assembled.resolution.merges,
                "duplicate_candidates": assembled.resolution.review,
                "relationship_flags": assembled.flags,
                "folded_relationships": assembled.folded,
                "last_sync": sync.as_dict(),
            },
        )
        self._write_json(self.settings.exports_dir / "knowledge_graph.json", assembled.as_dict())
        for document_id, parsed in layouts.items():
            self._write_json(self._layout_path(document_id), parsed.model_dump())
        for document_id, report in reports.items():
            self._write_json(self._validation_path(document_id), report.as_dict())
        return assembled, reports

    def graph(self) -> dict[str, list[dict]]:
        """The persisted graph, read from Neo4j."""
        return self.repository.load()

    def analysis(self) -> dict:
        path = self.settings.exports_dir / "analysis.json"
        if not path.exists():
            return {
                "conflicts": [],
                "successions": [],
                "observations": [],
                "merges": [],
                "duplicate_candidates": [],
                "relationship_flags": [],
                "folded_relationships": [],
                "last_sync": {},
            }
        return json.loads(path.read_text(encoding="utf-8"))

    def _record_validation(
        self, extractions: list[DocumentExtraction], reports: dict[int, ValidationReport]
    ) -> None:
        for extraction in extractions:
            if report := reports.get(extraction.document_id):
                self.provenance.record_validation(extraction, report.as_dict()["verdicts"])

    # --------------------------------------------------------------- retrieval
    def engine(self) -> RetrievalEngine:
        return RetrievalEngine(self.vector, self.repository, self.graph(), self.analysis())

    def query(
        self,
        question: str,
        mode: str,
        *,
        generate: bool = False,
        engine: RetrievalEngine | None = None,
        **options,
    ) -> tuple[RetrievalResult, str | None]:
        """Retrieve, optionally generate an answer, and log the query."""
        result = (engine or self.engine()).retrieve(question, mode, **options)
        answer = generate_answer(self.settings, result) if generate else None
        response = answer or (
            f"{len(result.evidence)} evidence item(s)"
            if result.sufficient
            else "Insufficient evidence"
        )
        sources = [
            {
                "document_id": document_id,
                "relevance_score": item.score,
                "source_type": item.kind,
                "evidence_reference": item.reference,
            }
            for item in result.evidence
            for document_id in item.document_ids
        ]
        self.registry.log_query(question, mode, response, result.latency_ms / 1000, sources)
        return result, answer

    # ------------------------------------------------------------------ helpers
    def _extraction_path(self, document_id: int) -> Path:
        return self.settings.extractions_dir / f"{document_id}.json"

    def _layout_path(self, document_id: int) -> Path:
        return self.settings.layout_dir / f"{document_id}.json"

    def _validation_path(self, document_id: int) -> Path:
        return self.settings.validation_dir / f"{document_id}.json"

    def extraction(self, document_id: int) -> DocumentExtraction | None:
        path = self._extraction_path(document_id)
        if not path.exists():
            return None
        return DocumentExtraction.model_validate_json(path.read_text(encoding="utf-8"))

    def _extractions(self) -> list[DocumentExtraction]:
        known = {d.id for d in self.registry.list_documents()}
        results = [self.extraction(document_id) for document_id in sorted(known)]
        return [r for r in results if r is not None]

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        """Atomic write so a crash never leaves a truncated artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        temporary.replace(path)
