"""Turn per-document extractions into one resolved knowledge graph.

Semantica's ``GraphBuilder`` produces the graph structure.  Entity merging is
deliberately **disabled** inside ``GraphBuilder`` (``merge_entities=False``): in
0.7.0 its default fuzzy resolver merges sibling names such as "Project Atlas"
and "Project Cobalt".  Resolution happens beforehand in
:mod:`backend.graph.resolution`, where every merge is explainable.

The application contributes the aggregation step: one edge per
(subject, predicate, object), carrying the evidence from every document that
states it, so nothing is overwritten.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from semantica.kg import GraphBuilder

from backend.graph.resolution import EntityResolution, ResolvedEntities
from backend.ontology.schema import ATTRIBUTES, FUNCTIONAL_RELATIONS, RELATION_BY_NAME
from backend.schemas import DocumentExtraction, Fact


@dataclass
class AssembledGraph:
    entities: list[dict]
    relationships: list[dict]
    observations: list[dict]  # single-valued claims, input to conflict detection
    resolution: ResolvedEntities
    unresolved_facts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """The ``{"entities", "relationships"}`` shape Semantica modules consume."""
        return {"entities": self.entities, "relationships": self.relationships}


def relationship_id(source: str, predicate: str, target: str, valid_from: str | None) -> str:
    raw = f"{source}|{predicate}|{target}|{valid_from or ''}"
    return "rel_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def assemble_graph(
    extractions: list[DocumentExtraction], resolution: EntityResolution | None = None
) -> AssembledGraph:
    resolver = resolution or EntityResolution()
    resolved = resolver.resolve(extractions)
    edges: dict[tuple[str, str, str, str | None], dict] = {}
    observations: list[dict] = []
    unresolved: list[str] = []

    for extraction in extractions:
        for fact in extraction.facts:
            subject_id = resolved.resolve(fact.subject_type, fact.subject)
            if subject_id is None:
                unresolved.append(fact.fact_id)
                continue
            if fact.kind == "attribute":
                value = fact.object
                entity_value = resolver.normalized_name(fact.object, "Technology")
                if fact.predicate == "primary_database":
                    value = entity_value
                observations.append(
                    _observation(extraction, fact, subject_id, resolved, fact.predicate, value)
                )
                continue

            object_id = resolved.resolve(fact.object_type, fact.object)
            if object_id is None or object_id == subject_id:
                unresolved.append(fact.fact_id)
                continue
            _add_evidence(edges, extraction, fact, subject_id, object_id)

            if fact.predicate in FUNCTIONAL_RELATIONS:
                property_name, object_type = FUNCTIONAL_RELATIONS[fact.predicate]
                if fact.object_type == object_type:
                    observations.append(
                        _observation(
                            extraction,
                            fact,
                            object_id,
                            resolved,
                            property_name,
                            resolved.entities[subject_id]["name"],
                        )
                    )

    entities = [dict(entity) for entity in resolved.entities.values()]
    _attach_attributes(entities, observations)
    relationships = [_finalise_edge(edge) for _, edge in sorted(edges.items(), key=_edge_order)]
    built = GraphBuilder(merge_entities=False, resolve_conflicts=False).build(
        {"entities": entities, "relationships": relationships}
    )
    return AssembledGraph(
        entities=built["entities"],
        relationships=built["relationships"],
        observations=observations,
        resolution=resolved,
        unresolved_facts=unresolved,
    )


def _edge_order(item: tuple) -> tuple[str, str, str, str]:
    (source, predicate, target, valid_from), _ = item
    return source, predicate, target, valid_from or ""


def _add_evidence(
    edges: dict, extraction: DocumentExtraction, fact: Fact, subject_id: str, object_id: str
) -> None:
    spec = RELATION_BY_NAME[fact.predicate]
    key = (subject_id, fact.predicate, object_id, fact.valid_from)
    reverse = (object_id, fact.predicate, subject_id, fact.valid_from)
    if spec.symmetric and reverse in edges:
        key = reverse
    edge = edges.setdefault(
        key,
        {"source": key[0], "target": key[2], "type": fact.predicate, "evidence": []},
    )
    edge["evidence"].append(
        {
            "fact_id": fact.fact_id,
            "document": extraction.filename,
            "document_id": extraction.document_id,
            "location": fact.location,
            "quote": fact.quote,
            "confidence": fact.confidence,
            "method": fact.method,
            "extracted_at": fact.extracted_at,
            "valid_from": fact.valid_from,
            "valid_until": fact.valid_until,
        }
    )


def _finalise_edge(edge: dict) -> dict:
    """Flatten evidence into parallel lists (Neo4j properties must be primitives)."""
    evidence = edge.pop("evidence")
    valid_from = next((e["valid_from"] for e in evidence if e["valid_from"]), None)
    valid_until = next((e["valid_until"] for e in evidence if e["valid_until"]), None)
    return {
        "id": relationship_id(edge["source"], edge["type"], edge["target"], valid_from),
        **edge,
        "confidence": max(e["confidence"] for e in evidence),
        "source_documents": sorted({e["document"] for e in evidence}),
        "document_ids": sorted({e["document_id"] for e in evidence}),
        "provenance_ids": [e["fact_id"] for e in evidence],
        "evidence_documents": [e["document"] for e in evidence],
        "evidence_locations": [e["location"] for e in evidence],
        "evidence_quotes": [e["quote"] for e in evidence],
        "extraction_method": evidence[0]["method"],
        "extracted_at": max(e["extracted_at"] for e in evidence),
        "valid_from": valid_from,
        "valid_until": valid_until,
    }


def _observation(
    extraction: DocumentExtraction,
    fact: Fact,
    entity_id: str,
    resolved: ResolvedEntities,
    property_name: str,
    value: str,
) -> dict:
    entity = resolved.entities[entity_id]
    return {
        "id": entity_id,
        "name": entity["name"],
        "type": entity["type"],
        "property": property_name,
        property_name: value,
        "value": value,
        "source": extraction.filename,
        "document_id": extraction.document_id,
        "section": fact.location,
        "quote": fact.quote,
        "confidence": fact.confidence,
        "fact_id": fact.fact_id,
        "valid_from": fact.valid_from,
        "valid_until": fact.valid_until,
    }


def _attach_attributes(entities: list[dict], observations: list[dict]) -> None:
    """Record every claimed value; set the scalar property only when undisputed."""
    attribute_names = {spec.name for spec in ATTRIBUTES}
    claims: dict[tuple[str, str], list[str]] = {}
    for observation in observations:
        if observation["property"] in attribute_names:
            values = claims.setdefault((observation["id"], observation["property"]), [])
            if observation["value"] not in values:
                values.append(observation["value"])
    by_id = {entity["id"]: entity for entity in entities}
    for (entity_id, name), values in claims.items():
        entity = by_id[entity_id]
        entity[f"{name}_values"] = sorted(values)
        if len(values) == 1:
            entity[name] = values[0]
