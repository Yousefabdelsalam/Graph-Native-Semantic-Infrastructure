"""End-to-end pipeline: incremental ingestion, provenance, persistence, failures."""

from __future__ import annotations

from sqlalchemy import select

from backend.services.lab import KnowledgeLab
from database.models import Query, QuerySource
from tests.conftest import make_settings
from tests.fakes import InMemoryRepository

FIRST = b"Acme AI developed VisionCore. VisionCore depends on PostgreSQL."
SECOND = b"FleetSense depends on Kafka."


def test_unchanged_content_is_not_processed_twice(lab) -> None:
    assert lab.ingest("a.txt", FIRST).status == "processed"
    assert lab.ingest("renamed.txt", FIRST).status == "skipped"
    assert lab.registry.counts() == {"documents": 1, "ingestion_runs": 1, "queries": 0}


def test_new_documents_do_not_remove_earlier_ones(lab) -> None:
    lab.ingest("a.txt", FIRST)
    chunks = lab.vector.count()
    lab.ingest("b.txt", SECOND)
    names = {e["name"] for e in lab.graph()["entities"]}
    assert {"VisionCore", "FleetSense", "PostgreSQL", "Kafka"} <= names
    assert lab.vector.count() > chunks


def test_reprocessing_does_not_duplicate(lab) -> None:
    outcome = lab.ingest("a.txt", FIRST)
    before = (lab.graph(), lab.vector.count())
    assert lab.process(outcome.document_id).status == "processed"
    after = (lab.graph(), lab.vector.count())
    assert [r["id"] for r in before[0]["relationships"]] == [
        r["id"] for r in after[0]["relationships"]
    ]
    assert before[1] == after[1]


def test_provenance_chain_survives_reprocessing(lab) -> None:
    outcome = lab.ingest("a.txt", FIRST)
    fact = lab.extraction(outcome.document_id).facts[0]
    lab.process(outcome.document_id)

    record = lab.provenance.get(fact.fact_id)
    assert record["source_document"] == "a.txt"
    assert record["source_quote"] == fact.quote
    assert record["source_location"] == fact.location
    lineage = lab.provenance.lineage(fact.fact_id)
    kinds = [entry["entity_type"] for entry in lineage["lineage_chain"]]
    assert "document" in kinds and kinds.count(fact.kind) >= 2  # earlier version kept
    assert lineage["integrity_verified"]
    assert lab.provenance.verify_integrity()["valid"]


def test_deleting_a_document_removes_data_but_keeps_audit_history(lab) -> None:
    first = lab.ingest("a.txt", FIRST)
    lab.ingest("b.txt", SECOND)
    fact_id = lab.extraction(first.document_id).facts[0].fact_id
    lab.delete_document(first.document_id)

    assert "VisionCore" not in {e["name"] for e in lab.graph()["entities"]}
    assert {c["filename"] for c in lab.vector.chunks()} == {"b.txt"}
    assert lab.provenance.get(fact_id)["invalidated"] is True


def test_malformed_upload_is_recorded_as_failed(lab) -> None:
    lab.ingest("good.txt", FIRST)
    outcome = lab.ingest("broken.pdf", b"%PDF-1.4 broken")
    assert outcome.status == "failed" and "parse" in outcome.message.lower()
    stored = lab.registry.get(outcome.document_id)
    assert stored.processing_status == "failed" and stored.error_message
    assert "VisionCore" in {e["name"] for e in lab.graph()["entities"]}


def test_graph_outage_leaves_other_stores_untouched_and_allows_retry(lab) -> None:
    lab.ingest("a.txt", FIRST)
    chunks = lab.vector.count()
    lab.repository.fail = True
    outcome = lab.ingest("b.txt", SECOND)
    assert outcome.status == "failed" and "Neo4j" in outcome.message
    assert lab.vector.count() == chunks
    assert lab.extraction(outcome.document_id) is None

    lab.repository.fail = False
    assert lab.ingest("b.txt", SECOND).status == "processed"  # same hash, retried


def test_state_persists_across_restart(tmp_path) -> None:
    settings = make_settings(tmp_path)
    repository = InMemoryRepository()  # stands in for the Neo4j server, which persists
    first = KnowledgeLab(settings, repository=repository)
    outcome = first.ingest("a.txt", FIRST)
    fact_id = first.extraction(outcome.document_id).facts[0].fact_id
    chunks = first.vector.chunks()
    conflicts = first.analysis()["conflicts"]
    first.close()

    second = KnowledgeLab(settings, repository=repository)
    try:
        assert second.registry.counts()["documents"] == 1
        assert second.vector.chunks() == chunks
        assert second.vector.search("What does VisionCore depend on?", 1)
        assert second.provenance.get(fact_id)["source_document"] == "a.txt"
        assert second.analysis()["conflicts"] == conflicts
        assert second.ingest("a.txt", FIRST).status == "skipped"
    finally:
        second.close()


def test_queries_and_their_sources_are_logged(lab) -> None:
    lab.ingest("a.txt", FIRST)
    lab.query("What does VisionCore depend on?", "hybrid")
    with lab.database.session() as session:
        query = session.scalars(select(Query)).one()
        sources = session.scalars(select(QuerySource)).all()
    assert query.retrieval_mode == "hybrid" and query.execution_time > 0
    assert {s.source_type for s in sources} >= {"chunk", "fact"}
