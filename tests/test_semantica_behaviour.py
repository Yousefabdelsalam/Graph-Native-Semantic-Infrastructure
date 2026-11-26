"""Characterisation tests: Semantica 0.7.0 behaviour this project works around.

A failure here after upgrading Semantica is good news or a warning: re-check the
corresponding workaround (see docs/semantica_integration.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdflib import Graph
from rdflib.namespace import OWL, RDF, RDFS
from semantica.kg import GraphBuilder
from semantica.parse import HTMLParser
from semantica.semantic_extract import NERExtractor
from semantica.vector_store import FAISSStore

from backend.ontology.schema import ALL_ENTITY_TYPES
from backend.ontology.service import OntologyService


def test_inner_product_scores_are_inverted() -> None:
    store = FAISSStore(dimension=2)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store.add_vectors(vectors, ids=["same", "orthogonal"], metric="inner_product")
    results = {r["id"]: r for r in store.search_similar(np.array([1.0, 0.0], np.float32), k=2)}
    assert results["same"]["distance"] > results["orthogonal"]["distance"]
    assert results["same"]["score"] < results["orthogonal"]["score"]  # the defect


def test_graph_builder_fuzzy_merge_joins_distinct_projects() -> None:
    graph = GraphBuilder(merge_entities=True, resolve_conflicts=False).build(
        {
            "entities": [
                {"id": "a", "name": "Project Atlas", "type": "Project"},
                {"id": "c", "name": "Project Cobalt", "type": "Project"},
            ],
            "relationships": [],
        }
    )
    assert len(graph["entities"]) == 1


def test_html_parser_collapses_line_breaks(tmp_path: Path) -> None:
    path = tmp_path / "x.html"
    path.write_text("<h2>Title</h2>\n<p>Body.</p>", encoding="utf-8")
    assert HTMLParser().parse(path).text == "Title Body."


def test_requested_method_silently_falls_back() -> None:
    entities = NERExtractor(method="regex").extract_entities(
        "Somebody Else arrived.", patterns={"Technology": r"(Redis)"}
    )
    assert entities
    assert all(e.metadata["extraction_method"] != "regex" for e in entities)


def test_shacl_generator_keeps_only_the_first_range() -> None:
    service = OntologyService()
    graph = {
        "entities": [
            {"id": "p", "name": "Ann", "type": "Person"},
            {"id": "j", "name": "Project X", "type": "Project"},
        ],
        # LEADS allows Team or Project.
        "relationships": [{"id": "r", "source": "p", "target": "j", "type": "LEADS"}],
    }
    assert not service.validate(graph, service.shacl())["conforms"]
    assert service.validate(graph, service.corrected_shacl())["conforms"]


def test_corrected_shapes_still_reject_wrong_types() -> None:
    service = OntologyService()
    graph = {
        "entities": [
            {"id": "p", "name": "Ann", "type": "Person"},
            {"id": "t", "name": "Redis", "type": "Technology"},
        ],
        "relationships": [{"id": "r", "source": "p", "target": "t", "type": "LEADS"}],
    }
    assert not service.validate(graph, service.corrected_shacl())["conforms"]


def _declared_and_referenced(turtle: str) -> tuple[set, set]:
    graph = Graph().parse(data=turtle, format="turtle")
    referenced = set(graph.objects(predicate=RDFS.domain)) | {
        o
        for s, o in graph.subject_objects(RDFS.range)
        if (s, RDF.type, OWL.ObjectProperty) in graph
    }
    return set(graph.subjects(RDF.type, OWL.Class)), referenced


def test_owl_writer_detaches_plain_names_from_declared_classes() -> None:
    service = OntologyService()
    native = str(service.engine.to_owl(service.ontology, format="turtle"))
    classes, referenced = _declared_and_referenced(native)
    assert referenced - classes  # the defect


def test_owl_domains_and_ranges_refer_to_declared_classes() -> None:
    classes, referenced = _declared_and_referenced(OntologyService().owl())
    assert len(classes) == len(ALL_ENTITY_TYPES)
    assert referenced and referenced <= classes
