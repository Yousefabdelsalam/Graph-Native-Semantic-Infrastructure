"""Decision intelligence: record a decision and trace it back to its sources.

Semantica:    ``AgentContext.record_decision`` writes a ``Decision`` node with
              ``ABOUT`` links to entity nodes in Neo4j, and
              ``trace_decision_explainability`` walks those links.
Application:  continues the trace from the linked entities to the facts between
              them and each fact's provenance (document, location, quote).

Known 0.7.0 limitation (reported, not hidden): ``find_precedents`` and
``query_decisions`` fail on a Neo4j ``GraphStore`` with ``KeyError:
'decision_id'`` because ``execute_query`` returns nodes as lists of property
names.
"""

from __future__ import annotations

from semantica.context import AgentContext

from backend.provenance.tracker import ProvenanceTracker
from backend.services.lab import KnowledgeLab
from backend.retrieval.vector import create_vector_store

SAMPLE_DECISION = {
    "category": "architecture",
    "scenario": "Select PostgreSQL as the primary database for VisionCore",
    "reasoning": (
        "VisionCore already depends on PostgreSQL according to the technology stack "
        "document. An archived wiki page names MongoDB instead; it is flagged as a conflict."
    ),
    "outcome": "PostgreSQL selected",
    "confidence": 0.8,
    "entities": ["product:visioncore", "technology:postgresql"],
}


class DecisionService:
    def __init__(self, lab: KnowledgeLab) -> None:
        self._lab = lab
        lab.repository.connect()
        # A separate in-memory vector store keeps decision memories out of the chunk index.
        memory_store, _ = create_vector_store(lab.settings.embedding_model)
        self.context = AgentContext(
            vector_store=memory_store,
            knowledge_graph=lab.repository.store,
            decision_tracking=True,
        )
        self._provenance: ProvenanceTracker = lab.provenance

    def record(
        self,
        *,
        category: str,
        scenario: str,
        reasoning: str,
        outcome: str,
        confidence: float,
        entities: list[str],
    ) -> str:
        known = {entity["id"] for entity in self._lab.graph()["entities"]}
        if missing := [e for e in entities if e not in known]:
            raise ValueError(f"Unknown entity id(s): {', '.join(missing)}")
        return self.context.record_decision(
            category=category,
            scenario=scenario,
            reasoning=reasoning,
            outcome=outcome,
            confidence=confidence,
            entities=entities,
        )

    def list_decisions(self) -> list[dict]:
        records = self._lab.repository.store.execute_query(
            "MATCH (d:Decision) RETURN properties(d) AS decision ORDER BY d.timestamp"
        )["records"]
        fields = ("decision_id", "category", "scenario", "reasoning", "outcome", "confidence")
        return [
            {
                **{name: r["decision"].get(name) for name in fields},
                "timestamp": str(r["decision"].get("timestamp")),
            }
            for r in records
        ]

    def trace(self, decision_id: str) -> dict:
        """Decision -> linked entities -> supporting facts -> source documents."""
        linked = [
            row["id"]
            for row in self._lab.repository.store.execute_query(
                "MATCH (:Decision {decision_id: $id})-[:ABOUT]->(e) RETURN DISTINCT e.id AS id",
                {"id": decision_id},
            )["records"]
        ]
        graph = self._lab.graph()
        names = {entity["id"]: entity["name"] for entity in graph["entities"]}
        facts = []
        # Facts that connect two linked entities, or all facts about a single one.
        for relationship in graph["relationships"]:
            ends = {relationship["source"], relationship["target"]}
            if not (ends <= set(linked) or (len(linked) == 1 and ends & set(linked))):
                continue
            for fact_id in relationship.get("provenance_ids", []):
                entry = self._provenance.get(fact_id) or {}
                facts.append(
                    {
                        "statement": f"{names[relationship['source']]} {relationship['type']} "
                        f"{names[relationship['target']]}",
                        "fact_id": fact_id,
                        "document": entry.get("source_document"),
                        "location": entry.get("source_location"),
                        "quote": entry.get("source_quote"),
                    }
                )
        conflicts = [c for c in self._lab.analysis()["conflicts"] if c["entity_id"] in linked]
        return {
            "decision_id": decision_id,
            "linked_entities": [names.get(i, i) for i in linked],
            "supporting_facts": facts,
            "open_conflicts": conflicts,
            "source_documents": sorted({f["document"] for f in facts if f["document"]}),
            "semantica_trace": self.context.trace_decision_explainability(decision_id),
        }

    def probe_precedent_search(self, scenario: str) -> str:
        """Run Semantica's precedent search and report what happens."""
        try:
            found = self.context.find_precedents(scenario, limit=3)
        except Exception as exc:
            return f"find_precedents failed: {type(exc).__name__}: {exc}"
        return f"find_precedents returned {len(found)} precedent(s)."
