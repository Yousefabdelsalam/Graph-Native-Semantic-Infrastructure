"""Entity resolution, graph construction and conflict detection."""

from __future__ import annotations

import pytest

from backend.graph.analytics import analyse
from backend.graph.builder import assemble_graph
from backend.graph.conflicts import detect_conflicts
from backend.graph.reasoning import infer, point_in_time
from backend.graph.resolution import clean_surface


@pytest.fixture(scope="module")
def assembled(sample_extractions):
    return assemble_graph(sample_extractions)


def _entity(assembled, entity_id: str) -> dict:
    return next(e for e in assembled.entities if e["id"] == entity_id)


def _edges(assembled) -> set[tuple[str, str, str]]:
    names = {e["id"]: e["name"] for e in assembled.entities}
    return {(names[r["source"]], r["type"], names[r["target"]]) for r in assembled.relationships}


def test_organisation_aliases_merge(assembled) -> None:
    acme = _entity(assembled, "organization:acme-ai")
    assert {"Acme AI Inc.", "Acme Artificial Intelligence"} <= set(acme["aliases"])
    assert len([e for e in assembled.entities if e["type"] == "Organization"]) == 2


def test_person_mentions_merge_across_documents(assembled) -> None:
    sarah = _entity(assembled, "person:sarah-chen")
    assert set(sarah["aliases"]) == {"Dr. Sarah Chen", "Sarah"}
    assert len(sarah["source_documents"]) >= 3


def test_ambiguous_first_name_is_not_merged(pipeline) -> None:
    from tests.conftest import make_document

    extraction = pipeline.extract(
        make_document(
            "Sarah Chen leads Project Atlas. Sarah Lee leads Project Beacon. "
            "Sarah leads the Platform Team."
        )
    )
    people = {e["name"] for e in assemble_graph([extraction]).entities if e["type"] == "Person"}
    assert people == {"Sarah Chen", "Sarah Lee", "Sarah"}


def test_similar_sibling_names_stay_separate_and_are_listed(assembled) -> None:
    projects = {e["name"] for e in assembled.entities if e["type"] == "Project"}
    assert {"Project Atlas", "Project Beacon", "Project Cobalt"} <= projects
    pairs = {frozenset((c["first"], c["second"])) for c in assembled.resolution.review}
    assert frozenset(("Project Atlas", "Project Cobalt")) in pairs


def test_technology_alias(assembled) -> None:
    assert ("DocuMind", "DEPENDS_ON", "PostgreSQL") in _edges(assembled)
    assert not any(e["name"] == "Postgres" for e in assembled.entities)


def test_same_fact_from_two_documents_is_one_edge_with_both_sources(assembled) -> None:
    edges = [
        r
        for r in assembled.relationships
        if r["type"] == "DEVELOPED" and r["target"] == "product:visioncore"
    ]
    assert len(edges) == 1
    assert edges[0]["source_documents"] == ["02_product_catalogue.pdf", "06_legacy_wiki.txt"]
    assert len(edges[0]["provenance_ids"]) == len(edges[0]["evidence_quotes"]) == 2


def test_every_edge_joins_known_entities(assembled) -> None:
    ids = {e["id"] for e in assembled.entities}
    assert all(r["source"] in ids and r["target"] in ids for r in assembled.relationships)
    assert assembled.unresolved_facts == []


def test_disputed_attributes_keep_every_value(assembled) -> None:
    vision = _entity(assembled, "product:visioncore")
    assert vision["launch_year_values"] == ["2023", "2024"]
    assert "launch_year" not in vision
    assert _entity(assembled, "product:documind")["launch_year"] == "2022"


def test_deliberate_contradictions_are_detected_with_sources(assembled) -> None:
    conflicts, successions = detect_conflicts(assembled.observations)
    found = {(c["entity_name"], c["property"]): c for c in conflicts}
    assert set(found) == {
        ("VisionCore", "launch_year"),
        ("VisionCore", "primary_database"),
        ("VisionCore Team", "lead"),
    }
    lead = found[("VisionCore Team", "lead")]
    assert {(c["value"], c["document"]) for c in lead["claims"]} == {
        ("Sarah Chen", "01_company_overview.txt"),
        ("David Okafor", "06_legacy_wiki.txt"),
    }
    assert all(s["tie"] for s in lead["suggestions"])
    assert [(s["entity_name"], s["property"]) for s in successions] == [("Platform Team", "lead")]


def test_source_credibility_changes_the_suggestion_only(assembled) -> None:
    conflicts, _ = detect_conflicts(assembled.observations, {"06_legacy_wiki.txt": 0.05})
    for conflict in conflicts:
        weighted = next(
            s for s in conflict["suggestions"] if s["strategy"] == "credibility_weighted"
        )
        assert not weighted["tie"]
        assert weighted["value"] in {"2023", "PostgreSQL", "Sarah Chen"}
        assert len(conflict["claims"]) == 2  # nothing removed


def test_agreeing_sources_are_not_a_conflict() -> None:
    claim = {
        "id": "product:x",
        "name": "X",
        "type": "Product",
        "property": "launch_year",
        "launch_year": "2020",
        "value": "2020",
        "section": "body",
        "quote": "q",
        "confidence": 0.7,
        "document_id": 1,
        "fact_id": "f",
    }
    assert detect_conflicts([{**claim, "source": "a"}, {**claim, "source": "b"}]) == ([], [])


def test_reasoner_derives_new_facts_without_storing_them(assembled) -> None:
    graph = assembled.as_dict()
    before = len(graph["relationships"])
    conclusions = {i["conclusion"] for i in infer(graph)}
    assert "DEPENDS_ON_INDIRECTLY(DocuMind, PostgreSQL)" in conclusions
    assert "ACCOUNTABLE_FOR(Priya Raman, Project Cobalt)" in conclusions
    assert len(graph["relationships"]) == before


@pytest.mark.parametrize(
    ("date", "expected"), [("2024-06-30", "marcus-webb"), ("2025-06-30", "priya-raman")]
)
def test_point_in_time_view(assembled, date: str, expected: str) -> None:
    leads = {
        r["source"]
        for r in point_in_time(assembled.as_dict(), date)["relationships"]
        if r["type"] == "LEADS" and r["target"] == "team:platform-team"
    }
    assert leads == {f"person:{expected}"}


def test_analytics_rank_the_hub_and_find_isolated_entities(assembled) -> None:
    result = analyse(assembled.as_dict())
    assert result["rankings"]["Degree"][0]["Entity"] == "Acme AI"
    assert result["rankings"]["PageRank"][0]["Entity"] == "Acme AI"
    assert ["MongoDB"] in result["components"]  # known only from an attribute claim


def test_clean_surface() -> None:
    assert clean_surface("Dr. Sarah Chen") == "Sarah Chen"
    assert clean_surface("Northwind Robotics Inc.") == "Northwind Robotics"
