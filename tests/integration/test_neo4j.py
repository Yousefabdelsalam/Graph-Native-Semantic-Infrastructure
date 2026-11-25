"""Graph persistence, traversal, pipeline and decisions on a real Neo4j."""

from __future__ import annotations

import copy

import pytest
from semantica.context import ContextRetriever

from backend.graph.builder import assemble_graph
from backend.graph.repository import GraphRepository
from backend.retrieval.vector import create_vector_store
from backend.services.decisions import SAMPLE_DECISION, DecisionService
from backend.services.lab import KnowledgeLab
from tests.conftest import make_settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def graph(sample_extractions) -> dict:
    return assemble_graph(sample_extractions).as_dict()


def _comparable(graph: dict) -> dict:
    ignored = {"created_at", "updated_at"}
    return {
        key: sorted(
            (
                {k: v for k, v in row.items() if k not in ignored and v not in (None, [])}
                for row in rows
            ),
            key=lambda row: row["id"],
        )
        for key, rows in graph.items()
        if key in ("entities", "relationships")
    }


def test_round_trip_preserves_the_graph(repository, graph) -> None:
    stats = repository.sync(graph)
    assert stats.nodes_created == len(graph["entities"])
    assert stats.relationships_created == len(graph["relationships"])
    assert _comparable(repository.load()) == _comparable(graph)


def test_second_sync_changes_nothing(repository, graph) -> None:
    repository.sync(graph)
    assert set(repository.sync(graph).as_dict().values()) == {0}


def test_sync_updates_in_place_and_removes_stale_data(repository, graph) -> None:
    repository.sync(graph)
    created = {e["id"]: e["created_at"] for e in repository.load()["entities"]}
    changed = copy.deepcopy(graph)
    changed["entities"][0]["mention_count"] += 1
    removed = changed["relationships"].pop()
    stats = repository.sync(changed)
    assert (stats.nodes_updated, stats.relationships_deleted, stats.nodes_created) == (1, 1, 0)
    loaded = repository.load()
    assert removed["id"] not in {r["id"] for r in loaded["relationships"]}
    assert {e["id"]: e["created_at"] for e in loaded["entities"]} == created


def test_graph_persists_for_a_new_connection(repository, neo4j_settings, graph) -> None:
    repository.sync(graph)
    fresh = GraphRepository(neo4j_settings)
    try:
        assert _comparable(fresh.load()) == _comparable(graph)
    finally:
        fresh.close()


def test_neighbourhood_traversal(repository, graph) -> None:
    repository.sync(graph)
    one = {e["id"] for e in repository.neighbors("product:documind", depth=1)}
    two = {e["id"] for e in repository.neighbors("product:documind", depth=2)}
    assert "product:visioncore" in one and "technology:redis" not in one
    assert one < two and "technology:redis" in two


def test_multi_hop_shortest_path(repository, graph) -> None:
    repository.sync(graph)
    path = repository.shortest_path("organization:northwind-robotics", "technology:postgresql")
    assert path["length"] == 3 == len(path["relationship_ids"])
    assert path["node_ids"][-1] == "technology:postgresql"
    assert repository.shortest_path("technology:mongodb", "technology:redis") is None


@pytest.fixture()
def neo4j_lab(repository, tmp_path, neo4j_settings):
    settings = make_settings(tmp_path).model_copy(update={"neo4j_uri": neo4j_settings.neo4j_uri})
    lab = KnowledgeLab(settings, repository=repository)
    assert all(o.status == "processed" for o in lab.load_sample_data())
    yield lab
    lab.database.dispose()


def test_pipeline_and_retrieval_on_neo4j(neo4j_lab) -> None:
    stats = neo4j_lab.repository.stats()
    assert stats["entities"] >= 25 and stats["relationships"] >= 45
    result, _ = neo4j_lab.query("How is Northwind Robotics connected to PostgreSQL?", "graph")
    assert result.paths and result.sufficient


def test_decision_is_traceable_to_sources_and_survives_sync(neo4j_lab) -> None:
    service = DecisionService(neo4j_lab)
    decision_id = service.record(**SAMPLE_DECISION)
    neo4j_lab.rebuild_graph()
    trace = service.trace(decision_id)
    assert set(trace["linked_entities"]) == {"VisionCore", "PostgreSQL"}
    assert trace["source_documents"] == ["03_technology_stack.docx"]
    assert all(f["quote"] for f in trace["supporting_facts"])
    assert any(c["property"] == "primary_database" for c in trace["open_conflicts"])
    assert [d["decision_id"] for d in service.list_decisions()] == [decision_id]
    with pytest.raises(ValueError, match="Unknown entity"):
        service.record(**{**SAMPLE_DECISION, "entities": ["product:nope"]})


# -- characterisation of Semantica 0.7.0 on Neo4j --------------------------------
def test_add_nodes_does_not_upsert(repository) -> None:
    node = {"id": "x", "type": "Thing", "properties": {"id": "x"}}
    repository.store.add_nodes([node])
    repository.store.add_nodes([node])
    assert len(repository.store.get_nodes(labels=["Thing"], limit=10)) == 2


def test_execute_query_flattens_nodes_to_property_names(repository) -> None:
    repository.store.create_node(["Thing"], {"id": "x", "name": "n"})
    record = repository.store.execute_query("MATCH (n:Thing) RETURN n")["records"][0]
    assert sorted(record["n"]) == ["id", "name"] and isinstance(record["n"], list)


def test_context_retriever_gets_no_graph_results_from_neo4j(repository, graph, neo4j_settings) -> None:
    repository.sync(graph)
    vectors, _ = create_vector_store(neo4j_settings.embedding_model)
    vectors.store_vectors([vectors.embed("seed")], metadata=[{"content": "seed"}], ids=["seed"])
    results = ContextRetriever(knowledge_graph=repository.store, vector_store=vectors).retrieve(
        "What does VisionCore depend on?"
    )
    assert [r.source for r in results] == ["vector:seed"]


def test_precedent_search_fails_on_neo4j(neo4j_lab) -> None:
    service = DecisionService(neo4j_lab)
    service.record(**SAMPLE_DECISION)
    assert "KeyError" in service.probe_precedent_search("choose a database")
