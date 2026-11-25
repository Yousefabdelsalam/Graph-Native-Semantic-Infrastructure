"""Fact provenance through Semantica's ``ProvenanceManager`` (W3C PROV-O, SQLite).

Chain recorded for every extracted statement::

    fact_<hash>  --derived from-->  document:<id>

``track_entity`` (not ``track_relationship``) is used for facts because, in
0.7.0, only it stores the source quote and keeps earlier versions when the same
id is tracked again - which is what re-processing a document does.
"""

from __future__ import annotations

import logging
from pathlib import Path

from semantica.provenance import ProvenanceManager

from backend.schemas import DocumentExtraction, ParsedDocument

logger = logging.getLogger(__name__)

AGENT = "semantica-knowledge-graph-lab"


class ProvenanceTracker:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.manager = ProvenanceManager(storage_path=str(path))

    def record_document(self, document: ParsedDocument, file_hash: str) -> None:
        self.manager.track_entity(
            f"document:{document.document_id}",
            source=document.filename,
            entity_type="document",
            activity_id="ingestion",
            agent_id=AGENT,
            metadata={
                "document_id": document.document_id,
                "file_hash": file_hash,
                "parser": document.parser,
                "segments": [s.label for s in document.segments],
            },
        )

    def record_extraction(self, extraction: DocumentExtraction) -> int:
        recorded = 0
        for fact in extraction.facts:
            entry = self.manager.track_entity(
                fact.fact_id,
                source=extraction.filename,
                entity_type=fact.kind,
                parent_entity_id=f"document:{extraction.document_id}",
                source_location=fact.location,
                source_quote=fact.quote,
                confidence=fact.confidence,
                activity_id=f"extraction:{fact.method}",
                agent_id=AGENT,
                valid_from=fact.valid_from,
                valid_until=fact.valid_until,
                metadata={
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.object,
                    "document_id": extraction.document_id,
                    "extraction_method": fact.method,
                    "extracted_at": fact.extracted_at,
                },
            )
            recorded += entry is not None
        if recorded != len(extraction.facts):
            raise RuntimeError(
                f"Provenance storage accepted {recorded} of {len(extraction.facts)} facts."
            )
        return recorded

    def invalidate(self, fact_ids: list[str], reason: str) -> int:
        """Mark facts that are no longer supported (never delete audit history)."""
        invalidated = 0
        for fact_id in fact_ids:
            if self.manager.get_provenance(fact_id) is None:
                continue
            self.manager.invalidate(fact_id, agent_id=AGENT, reason=reason)
            invalidated += 1
        return invalidated

    def get(self, fact_id: str) -> dict | None:
        return self.manager.get_provenance(fact_id)

    def lineage(self, fact_id: str) -> dict:
        """Full chain: current entry, earlier versions and the source document."""
        return self.manager.get_lineage(fact_id)

    def verify_integrity(self) -> dict:
        """Hash-chain check provided by Semantica."""
        return self.manager.verify_chain()

    def statistics(self) -> dict:
        return self.manager.get_statistics()

    def export_prov_turtle(self) -> str:
        return self.manager.export_prov()
