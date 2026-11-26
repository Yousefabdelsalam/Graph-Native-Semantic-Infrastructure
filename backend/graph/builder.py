"""Turn per-document extractions into one resolved knowledge graph.

Semantica's ``GraphBuilder`` produces the graph structure.  Entity merging is
deliberately **disabled** inside ``GraphBuilder`` (``merge_entities=False``): in
0.7.0 its default fuzzy resolver merges sibling names such as "Project Atlas"
and "Project Cobalt".  Resolution happens beforehand in
:mod:`backend.graph.resolution`, where every merge is explainable.

The application contributes the aggregation step: one edge per
(subject, predicate, object), carrying the evidence from every document that
states it, so nothing is overwritten.

Every edge has a ``support_status``: ``supported`` when at least one piece of
evidence states it directly, otherwise ``inferred``.  Rejected facts never reach
this module (see :mod:`backend.extraction.validation`).  Facts extracted by a
model must have been validated; the rule methods only produce facts from a single
sentence naming both ends.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from semantica.kg import GraphBuilder

from backend.extraction.validation import INFERRED, SUPPORTED, relationship_flags
from backend.graph.resolution import EntityResolution, ResolvedEntities
from backend.ontology.schema import ALL_RELATION_BY_NAME, ATTRIBUTES, FUNCTIONAL_RELATIONS
from backend.schemas import DocumentExtraction, Fact


@dataclass
class AssembledGraph:
    entities: list[dict]
    relationships: list[dict]
    observations: list[dict]  # single-valued claims, input to conflict detection
    resolution: ResolvedEntities
    unresolved_facts: list[str] = field(default_factory=list)
    flags: list[dict] = field(default_factory=list)  # reversed duplicates, conflicts
    folded: list[dict] = field(default_factory=list)  # vague links merged into specific ones

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

            # Only directly stated facts may take part in conflict detection.
            if fact.predicate in FUNCTIONAL_RELATIONS and _support(fact) == SUPPORTED:
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
    folded = _fold_vague_edges(edges)
    relationships = [_finalise_edge(edge) for _, edge in sorted(edges.items(), key=_edge_order)]
    flags = relationship_flags(
        [(r["id"], r["source"], r["type"], r["target"]) for r in relationships]
    )
    by_relationship = {r["id"]: r for r in relationships}
    for flag in flags:
        for identifier in flag["ids"]:
            by_relationship[identifier].setdefault("review_flags", []).append(flag["detail"])
    built = GraphBuilder(merge_entities=False, resolve_conflicts=False).build(
        {"entities": entities, "relationships": relationships}
    )
    return AssembledGraph(
        entities=built["entities"],
        relationships=built["relationships"],
        observations=observations,
        resolution=resolved,
        unresolved_facts=unresolved,
        flags=flags,
        folded=folded,
    )


def _support(fact: Fact) -> str:
    if fact.support:
        return fact.support
    if fact.method == "llm":
        raise ValueError(
            f"Model-extracted fact {fact.fact_id} has not been validated against the document."
        )
    return SUPPORTED


def _fold_vague_edges(edges: dict) -> list[dict]:
    """Merge RELATED_TO into a specific link between the same two entities.

    Only when the specific link is at least as well supported, so that a directly
    stated "related to" never disappears behind an inferred claim.
    """
    folded: list[dict] = []
    for key in sorted((k for k in edges if k[1] == "RELATED_TO"), key=str):
        ends = {key[0], key[2]}
        vague = edges[key]
        for other_key, other in sorted(edges.items(), key=lambda item: str(item[0])):
            if other_key[1] == "RELATED_TO" or {other_key[0], other_key[2]} != ends:
                continue
            if _edge_status(other) == INFERRED and _edge_status(vague) == SUPPORTED:
                continue
            for evidence in vague["evidence"]:
                other["evidence"].append({**evidence, "folded": True})
                folded.append(
                    {
                        "fact_id": evidence["fact_id"],
                        "from": [key[0], "RELATED_TO", key[2]],
                        "into": [other_key[0], other_key[1], other_key[2]],
                    }
                )
            del edges[key]
            break
    return folded


def _edge_status(edge: dict) -> str:
    own = [e for e in edge["evidence"] if not e.get("folded")]
    return SUPPORTED if any(e["support"] == SUPPORTED for e in own) else INFERRED


def _edge_order(item: tuple) -> tuple[str, str, str, str]:
    (source, predicate, target, valid_from), _ = item
    return source, predicate, target, valid_from or ""


def _add_evidence(
    edges: dict, extraction: DocumentExtraction, fact: Fact, subject_id: str, object_id: str
) -> None:
    spec = ALL_RELATION_BY_NAME[fact.predicate]
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
            "support": _support(fact),
        }
    )


def _finalise_edge(edge: dict) -> dict:
    """Flatten evidence into parallel lists (Neo4j properties must be primitives)."""
    status = _edge_status(edge)
    # Directly stated evidence first.
    evidence = sorted(edge.pop("evidence"), key=lambda e: e["support"] != SUPPORTED)
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
        "evidence_support": [e["support"] for e in evidence],
        "support_status": status,
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
