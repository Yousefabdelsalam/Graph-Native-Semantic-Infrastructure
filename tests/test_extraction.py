"""Entity, relationship, attribute and temporal extraction."""

from __future__ import annotations

import pytest

from backend.extraction.extractor import ExtractionError, ExtractionPipeline
from tests.conftest import make_document

EXAMPLE = (
    "Acme AI developed VisionCore, a computer vision platform. "
    "Sarah leads the VisionCore Team. VisionCore depends on PostgreSQL."
)


def _facts(extraction) -> set[tuple[str, str, str]]:
    return {(f.subject, f.predicate, f.object) for f in extraction.facts}


def test_brief_example_yields_the_expected_facts(pipeline) -> None:
    extraction = pipeline.extract(make_document(EXAMPLE))
    assert _facts(extraction) == {
        ("Acme AI", "DEVELOPED", "VisionCore"),
        ("Sarah", "LEADS", "VisionCore Team"),
        ("VisionCore", "DEPENDS_ON", "PostgreSQL"),
    }
    types = {(m.text, m.entity_type) for m in extraction.mentions}
    assert {
        ("Acme AI", "Organization"),
        ("VisionCore", "Product"),
        ("Sarah", "Person"),
        ("VisionCore Team", "Team"),
        ("PostgreSQL", "Technology"),
    } <= types


def test_every_fact_points_back_to_its_sentence(pipeline) -> None:
    document = make_document(EXAMPLE)
    for fact in pipeline.extract(document).facts:
        assert document.text[fact.start : fact.end] == fact.quote
        assert fact.subject.split()[-1] in fact.quote
        assert fact.location.startswith("body, chars ")
        assert fact.method == "regex" and 0 < fact.confidence <= 1
        assert fact.extracted_at


def test_passive_voice_is_oriented_correctly(pipeline) -> None:
    extraction = pipeline.extract(make_document("Project Atlas is owned by the VisionCore Team."))
    assert _facts(extraction) == {("VisionCore Team", "OWNS", "Project Atlas")}


def test_attributes_and_validity_periods(pipeline) -> None:
    extraction = pipeline.extract(
        make_document(
            "VisionCore was launched in 2023.\n"
            "The primary database of VisionCore is MongoDB.\n"
            "Marcus Webb led the Platform Team from 2023-01-01 until 2024-12-31."
        )
    )
    by_predicate = {f.predicate: f for f in extraction.facts}
    assert by_predicate["launch_year"].object == "2023"
    assert by_predicate["launch_year"].kind == "attribute"
    assert by_predicate["primary_database"].object == "MongoDB"
    lead = by_predicate["LEADS"]
    assert (lead.valid_from, lead.valid_until) == ("2023-01-01", "2024-12-31")


def test_relations_do_not_cross_sentences(pipeline) -> None:
    extraction = pipeline.extract(make_document("We love VisionCore. Redis is fast. It uses\nKafka."))
    assert extraction.facts == []


def test_library_fallback_results_are_kept_out(pipeline) -> None:
    extraction = pipeline.extract(make_document("Her Deputy arrived. Nothing Else happened."))
    assert extraction.mentions == []
    assert extraction.facts == []
    assert any("fallback" in reason for reason in extraction.discarded)


def test_domain_and_range_are_enforced(pipeline) -> None:
    extraction = pipeline.extract(make_document("PostgreSQL depends on VisionCore."))
    assert extraction.facts == []
    assert extraction.discarded.get("domain/range violation: DEPENDS_ON") == 1


def test_native_pattern_baseline_finds_no_domain_facts(settings) -> None:
    baseline = ExtractionPipeline(settings, "pattern").extract(make_document(EXAMPLE))
    assert baseline.facts == []


def test_triplets_come_from_semantica(pipeline) -> None:
    extraction = pipeline.extract(make_document(EXAMPLE))
    assert ["VisionCore", "DEPENDS_ON", "PostgreSQL"] in extraction.triplets


def test_llm_method_requires_a_key(settings) -> None:
    with pytest.raises(ExtractionError, match="GROQ_API_KEY"):
        ExtractionPipeline(settings, "llm")
