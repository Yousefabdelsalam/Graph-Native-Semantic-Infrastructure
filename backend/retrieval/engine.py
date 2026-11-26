"""The retrieval modes compared by this lab.

=============  =================================================================
mode           what runs
=============  =================================================================
``vector``     FAISS similarity search over chunks (Semantica ``VectorStore``).
``graph``      Application query planner: link entities named in the question
               (falling back to Semantica ``ContextRetriever`` entity matches),
               expand with ``GraphStore.get_neighbors`` / ``shortest_path``,
               filter by time with ``TemporalGraphQuery``, then rank the facts.
``hybrid``     ``graph`` facts plus ``vector`` chunks; entities mentioned in the
               top chunks seed the traversal when the question names none.
``semantica``  Semantica ``ContextRetriever`` exactly as shipped (vector + graph
               merge, ``hybrid_alpha=0.5``).  The control for "native" results.
=============  =================================================================

Relationships whose ``support_status`` is ``inferred`` (derived from document
structure) are always labelled, ranked slightly lower and never make graph
evidence sufficient on their own.

Semantica 0.7.0 note: ``ContextRetriever`` cannot be pointed at a Neo4j
``GraphStore`` for natural-language questions (it passes the question to
``GraphStore.query`` as Cypher and the error is swallowed), so it receives the
``{"entities", "relationships"}`` dictionary loaded from Neo4j instead.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from itertools import combinations

from semantica.context import ContextRetriever
from semantica.kg import TemporalGraphQuery

from backend.graph.repository import GraphRepository
from backend.retrieval.vector import VectorIndex
from backend.schemas import Evidence, RetrievalResult

logger = logging.getLogger(__name__)

MODES: dict[str, str] = {
    "vector": "Traditional Vector Search",
    "graph": "Graph Retrieval",
    "hybrid": "Hybrid GraphRAG",
    "semantica": "Semantica ContextRetriever (native)",
}

# Calibrated on the sample corpus with bge-small-en-v1.5; see docs/evaluation_results.md.
VECTOR_SUFFICIENCY_THRESHOLD = 0.62
GRAPH_FACT_THRESHOLD = 0.68
SEMANTIC_SEEDS = 4

_HOP_BONUS = {0: 0.15, 1: 0.06}
_PATH_BONUS = 0.30
_INFERRED_PENALTY = 0.05  # ranks a stated fact above an equally relevant inferred one
INFERRED_LABEL = "INFERRED (from document structure, not confirmed)"
ONLY_INFERRED_NOTE = (
    "Only inferred relationships were found. They follow from document structure and "
    "are not confirmed facts."
)
SOME_INFERRED_NOTE = "Facts labelled INFERRED are not confirmed."
_YEAR_SCOPE = re.compile(r"\b(?:in|during|as of)\s+((?:19|20)\d{2})\b", re.IGNORECASE)
_CONFLICT_INTENT = re.compile(r"conflict|contradict|disagree|inconsisten", re.IGNORECASE)
_PROPER_NOUNS = re.compile(r"(?<!^)\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*")


class _EmbeddingOnly:
    """Gives ContextRetriever embeddings for graph matching but no chunk search."""

    def __init__(self, index: VectorIndex) -> None:
        self._index = index

    def embed(self, text: str):
        return self._index.embed(text)


def _is_inferred(relationship: dict) -> bool:
    return relationship.get("support_status") == "inferred"


class RetrievalEngine:
    def __init__(
        self,
        vector_index: VectorIndex,
        repository: GraphRepository,
        graph: dict[str, list[dict]],
        analysis: dict,
    ) -> None:
        self.vector = vector_index
        self.repository = repository
        self.graph = graph
        self.conflicts: list[dict] = analysis.get("conflicts", [])
        self.observations: list[dict] = analysis.get("observations", [])
        self.entities = {entity["id"]: entity for entity in graph["entities"]}
        self.relationships = {rel["id"]: rel for rel in graph["relationships"]}
        self._temporal = TemporalGraphQuery()

    # ------------------------------------------------------------------ public
    def retrieve(
        self,
        question: str,
        mode: str,
        *,
        top_k: int = 4,
        max_facts: int = 12,
        max_hops: int = 3,
        as_of: str | None = None,
    ) -> RetrievalResult:
        if mode not in MODES:
            raise ValueError(f"Unknown retrieval mode '{mode}'.")
        question = question.strip()
        if not question:
            raise ValueError("The question is empty.")
        started = time.perf_counter()
        if mode == "vector":
            result = self._vector(question, top_k)
        elif mode == "graph":
            result = self._graph(question, max_facts, max_hops, as_of)
        elif mode == "hybrid":
            result = self._hybrid(question, top_k, max_facts, max_hops, as_of)
        else:
            result = self._native(question, top_k, max_hops)
        result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    # ------------------------------------------------------------ vector mode
    def _vector(self, question: str, top_k: int) -> RetrievalResult:
        evidence = self.vector.search(question, top_k)
        best = max((e.score or 0.0 for e in evidence), default=0.0)
        sufficient = best >= VECTOR_SUFFICIENCY_THRESHOLD
        note = "" if sufficient else (
            f"Best chunk similarity {best:.2f} is below {VECTOR_SUFFICIENCY_THRESHOLD}: "
            "the corpus probably does not answer this."
        )
        return RetrievalResult(
            question=question,
            mode="vector",
            latency_ms=0,
            evidence=evidence,
            sufficient=sufficient,
            note=note,
        )

    # ------------------------------------------------------------- graph mode
    def link_entities(self, text: str) -> list[str]:
        """Entities whose name or alias appears in the text (longest match wins)."""
        names: list[tuple[str, str]] = []
        for entity in self.entities.values():
            for name in [entity["name"], *entity.get("aliases", [])]:
                names.append((name, entity["id"]))
        taken: list[tuple[int, int]] = []
        found: dict[str, int] = {}
        for name, entity_id in sorted(names, key=lambda n: -len(n[0])):
            for match in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE):
                span = match.span()
                if any(span[0] < end and start < span[1] for start, end in taken):
                    continue
                taken.append(span)
                found.setdefault(entity_id, span[0])
        return sorted(found, key=found.get)

    def _semantic_seeds(self, question: str, max_hops: int) -> list[str]:
        retriever = ContextRetriever(
            knowledge_graph=self.graph,
            vector_store=_EmbeddingOnly(self.vector),
            hybrid_alpha=1.0,
            max_expansion_hops=max_hops,
        )
        results = retriever.retrieve(question, max_results=SEMANTIC_SEEDS)
        return [r.metadata["node_id"] for r in results if r.metadata.get("node_id")]

    def _distances(self, seeds: list[str], max_hops: int) -> dict[str, int]:
        """Hop distance from the nearest seed, using Neo4j neighbourhood queries."""
        distances = {seed: 0 for seed in seeds}
        for depth in range(1, max_hops + 1):
            for seed in seeds:
                for entity in self.repository.neighbors(seed, depth=depth):
                    distances.setdefault(entity["id"], depth)
        return distances

    def _valid_relationships(self, as_of: str | None) -> dict[str, dict]:
        if not as_of:
            return self.relationships
        snapshot = self._temporal.query_at_time(self.graph, "", at_time=as_of)
        return {rel["id"]: rel for rel in snapshot["relationships"]}

    def _sentence(self, relationship: dict) -> str:
        source = self.entities[relationship["source"]]["name"]
        target = self.entities[relationship["target"]]["name"]
        return f"{source} {relationship['type'].replace('_', ' ').lower()} {target}"

    def _statement(self, relationship: dict) -> str:
        source = self.entities[relationship["source"]]["name"]
        target = self.entities[relationship["target"]]["name"]
        window = ""
        if relationship.get("valid_from") or relationship.get("valid_until"):
            window = (
                f" [valid {relationship.get('valid_from') or '…'} → "
                f"{relationship.get('valid_until') or 'now'}]"
            )
        quotes = relationship.get("evidence_quotes", [])
        support = relationship.get("evidence_support") or [None] * len(quotes)
        if not _is_inferred(relationship):
            quotes = [
                f"{quote} (inferred)" if status == "inferred" else quote
                for quote, status in zip(quotes, support)
            ]
        joined = " | ".join(dict.fromkeys(quotes))
        label = f"{INFERRED_LABEL}: " if _is_inferred(relationship) else ""
        return f"{label}{source} {relationship['type']} {target}{window} — “{joined}”"

    def _graph(
        self,
        question: str,
        max_facts: int,
        max_hops: int,
        as_of: str | None,
        extra_seeds: list[str] | None = None,
    ) -> RetrievalResult:
        named = self.link_entities(question)
        seeds = named or extra_seeds or []
        semantic = False
        if not seeds:
            seeds, semantic = self._semantic_seeds(question, max_hops), True
        if as_of is None and (scope := _YEAR_SCOPE.search(question)):
            as_of = f"{scope.group(1)}-06-30"
        valid = self._valid_relationships(as_of)
        distances = self._distances(seeds, max_hops)

        paths: list[str] = []
        on_path: set[str] = set()
        for first, second in combinations(named, 2):
            path = self.repository.shortest_path(first, second, max_depth=max_hops + 2)
            if path and all(rel in valid for rel in path["relationship_ids"]):
                on_path.update(path["relationship_ids"])
                paths.append(self._describe_path(path))

        candidates = [
            rel
            for rel_id, rel in valid.items()
            if rel_id in on_path or (rel["source"] in distances and rel["target"] in distances)
        ]
        similarities = self.vector.similarities(question, [self._sentence(r) for r in candidates])
        ranked = []
        for relationship, similarity in zip(candidates, similarities):
            hop = min(
                distances.get(relationship["source"], 99), distances.get(relationship["target"], 99)
            )
            score = similarity + _HOP_BONUS.get(hop, 0.0)
            if relationship["id"] in on_path:
                score += _PATH_BONUS
            if _is_inferred(relationship):
                score -= _INFERRED_PENALTY
            ranked.append((score, similarity, relationship))
        ranked.sort(key=lambda item: (-item[0], item[2]["id"]))
        selected = ranked[:max_facts]

        evidence = self._conflict_evidence(question, named)
        evidence += self._attribute_evidence(named)
        evidence += [self._fact_evidence(rel, score) for score, _, rel in selected]

        used = {r["source"] for _, _, r in selected} | {r["target"] for _, _, r in selected}
        used |= set(named)
        sufficient, note = self._graph_sufficiency(
            question,
            named,
            [s for _, s, rel in selected if not _is_inferred(rel)],
            bool(evidence),
        )
        if semantic and sufficient:
            note = "No known entity was named; traversal started from semantic matches."
        if evidence and all(item.support == "inferred" for item in evidence):
            sufficient, note = False, ONLY_INFERRED_NOTE
        elif any(item.support == "inferred" for item in evidence):
            note = f"{note} {SOME_INFERRED_NOTE}".strip()
        if as_of:
            note = f"{note} Facts filtered to those valid on {as_of}.".strip()
        return RetrievalResult(
            question=question,
            mode="graph",
            latency_ms=0,
            entities=[self.entities[i] for i in sorted(used)],
            relationships=[rel for _, _, rel in selected],
            paths=paths,
            evidence=evidence,
            sufficient=sufficient,
            note=note,
        )

    def _graph_sufficiency(
        self, question: str, named: list[str], similarities: list[float], has_evidence: bool
    ) -> tuple[bool, str]:
        if not has_evidence:
            return False, "No graph facts are connected to this question."
        if named:
            return True, ""
        unknown = [m.group(0) for m in _PROPER_NOUNS.finditer(question)]
        if unknown:
            return False, (
                f"The graph has no entity called {', '.join(repr(u) for u in unknown)}; "
                "the facts shown are only semantically similar."
            )
        best = max(similarities, default=0.0)
        if best < GRAPH_FACT_THRESHOLD:
            return False, f"No fact is close enough to the question (best {best:.2f})."
        return True, ""

    def _fact_evidence(self, relationship: dict, score: float) -> Evidence:
        locations = [
            f"{document} ({location})"
            for document, location in zip(
                relationship.get("evidence_documents", []),
                relationship.get("evidence_locations", []),
            )
        ]
        return Evidence(
            kind="fact",
            text=self._statement(relationship),
            score=round(score, 4),
            document_ids=list(relationship.get("document_ids", [])),
            filenames=list(relationship.get("source_documents", [])),
            location="; ".join(locations),
            reference=relationship["id"],
            support="inferred" if _is_inferred(relationship) else "supported",
        )

    def _conflict_evidence(self, question: str, named: list[str]) -> list[Evidence]:
        """Surface disputes about the entities asked about."""
        asks = bool(_CONFLICT_INTENT.search(question))
        relevant = [
            c for c in self.conflicts if c["entity_id"] in named or (asks and not named)
        ]
        if asks and named:
            # Include disputes on neighbours, e.g. the lead of "<product> Team".
            near = {i for n in named for i in self._adjacent(n)} | set(named)
            relevant = [c for c in self.conflicts if c["entity_id"] in near]
        evidence = []
        for conflict in relevant:
            claims = "; ".join(
                f"{c['value']} per {c['document']} (“{c['quote']}”)" for c in conflict["claims"]
            )
            evidence.append(
                Evidence(
                    kind="conflict",
                    text=(
                        f"CONFLICT: {conflict['entity_name']}.{conflict['property']} is "
                        f"disputed — {claims}"
                    ),
                    document_ids=sorted({c["document_id"] for c in conflict["claims"]}),
                    filenames=sorted({c["document"] for c in conflict["claims"]}),
                    location="; ".join(f"{c['document']} ({c['location']})" for c in conflict["claims"]),
                    reference=conflict["conflict_id"],
                )
            )
        return evidence

    def _attribute_evidence(self, named: list[str]) -> list[Evidence]:
        """Single-valued claims (launch year, primary database) about named entities."""
        disputed = {(c["entity_id"], c["property"]) for c in self.conflicts}
        grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for o in self.observations:
            if o["id"] in named and o["property"] != "lead" and (o["id"], o["property"]) not in disputed:
                grouped[(o["id"], o["property"], o["value"])].append(o)
        return [
            Evidence(
                kind="fact",
                text=f"{claims[0]['name']} {prop} = {value} — “{claims[0]['quote']}”",
                document_ids=sorted({c["document_id"] for c in claims}),
                filenames=sorted({c["source"] for c in claims}),
                location="; ".join(f"{c['source']} ({c['section']})" for c in claims),
                reference=claims[0]["fact_id"],
            )
            for (_, prop, value), claims in sorted(grouped.items())
        ]

    def _adjacent(self, entity_id: str) -> set[str]:
        return {
            other
            for rel in self.relationships.values()
            if entity_id in (rel["source"], rel["target"])
            for other in (rel["source"], rel["target"])
        }

    def _describe_path(self, path: dict) -> str:
        parts = [self.entities[path["node_ids"][0]]["name"]]
        for index, rel_id in enumerate(path["relationship_ids"]):
            relationship = self.relationships[rel_id]
            forward = relationship["source"] == path["node_ids"][index]
            kind = relationship["type"] + (" (inferred)" if _is_inferred(relationship) else "")
            arrow = f"-[{kind}]->" if forward else f"<-[{kind}]-"
            parts += [arrow, self.entities[path["node_ids"][index + 1]]["name"]]
        return " ".join(parts)

    # ------------------------------------------------------------ hybrid mode
    def _hybrid(
        self, question: str, top_k: int, max_facts: int, max_hops: int, as_of: str | None
    ) -> RetrievalResult:
        chunks = self.vector.search(question, top_k)
        chunk_seeds: list[str] = []
        if not self.link_entities(question):
            for chunk in chunks[:2]:
                chunk_seeds += [e for e in self.link_entities(chunk.text) if e not in chunk_seeds]
        graph = self._graph(question, max_facts, max_hops, as_of, extra_seeds=chunk_seeds[:6])
        best_chunk = max((c.score or 0.0 for c in chunks), default=0.0)
        named = self.link_entities(question)
        sufficient = graph.sufficient and (bool(named) or best_chunk >= VECTOR_SUFFICIENCY_THRESHOLD)
        note = graph.note
        # Document text can answer even when the graph only holds inferred links.
        if note.startswith(ONLY_INFERRED_NOTE) and best_chunk >= VECTOR_SUFFICIENCY_THRESHOLD:
            sufficient = True
            note = f"{note} The answer must rest on the document passages."
        if not sufficient and not note:
            note = f"Neither graph facts nor chunks (best {best_chunk:.2f}) are convincing."
        return graph.model_copy(
            update={
                "mode": "hybrid",
                "evidence": [*graph.evidence, *chunks],
                "sufficient": sufficient,
                "note": note,
            }
        )

    # ------------------------------------------------------------ native mode
    def _native(self, question: str, top_k: int, max_hops: int) -> RetrievalResult:
        retriever = ContextRetriever(
            knowledge_graph=self.graph,
            vector_store=self.vector.store,
            hybrid_alpha=0.5,
            max_expansion_hops=max_hops,
        )
        results = retriever.retrieve(question, max_results=top_k * 2) if self.vector.count() else []
        evidence: list[Evidence] = []
        entities: dict[str, dict] = {}
        relationships: dict[str, dict] = {}
        for item in results:
            metadata = item.metadata or {}
            if item.source and item.source.startswith("vector:"):
                evidence.append(
                    Evidence(
                        kind="chunk",
                        text=item.content,
                        score=round(float(item.score), 4),
                        document_ids=[metadata["document_id"]],
                        filenames=[metadata["filename"]],
                        location=metadata.get("location"),
                        reference=metadata.get("chunk_id", item.source),
                    )
                )
                continue
            related = item.related_relationships or []
            for rel in related:
                relationships[rel["id"]] = rel
            node_id = metadata.get("node_id")
            if node_id in self.entities:
                entities[node_id] = self.entities[node_id]
            # The native result does not know about evidence status; add it.
            unconfirmed = [
                self._sentence(self.relationships[r["id"]])
                for r in related
                if r.get("id") in self.relationships and _is_inferred(self.relationships[r["id"]])
            ]
            text = item.content
            if unconfirmed:
                text = f"{text}\n{INFERRED_LABEL}: {'; '.join(unconfirmed)}"
            evidence.append(
                Evidence(
                    kind="context",
                    support="inferred" if related and len(unconfirmed) == len(related) else None,
                    text=text,
                    score=round(float(item.score), 4),
                    document_ids=sorted({d for r in related for d in r.get("document_ids", [])}),
                    filenames=sorted({d for r in related for d in r.get("source_documents", [])}),
                    location=None,  # the native result carries no source location
                    reference=item.source or "",
                )
            )
        return RetrievalResult(
            question=question,
            mode="semantica",
            latency_ms=0,
            entities=list(entities.values()),
            relationships=list(relationships.values()),
            evidence=evidence,
            sufficient=any(item.support != "inferred" for item in evidence),
            note="ContextRetriever has no insufficiency signal; any result counts as sufficient, "
            "except results resting only on inferred relationships.",
        )
