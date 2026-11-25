"""Entity, relationship, triplet and event extraction through Semantica.

Division of labour
------------------
Semantica:    ``NERExtractor``, ``RelationExtractor``, ``TripletExtractor`` and
              ``EventDetector`` do all matching / model calls.
Application:  supplies the domain patterns, runs extraction sentence by sentence
              (for exact provenance), resolves overlapping spans, enforces the
              ontology's domain/range, and records what was discarded and why.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from semantica.semantic_extract import (
    EventDetector,
    NERExtractor,
    RelationExtractor,
    TripletExtractor,
)

from backend.config import Settings
from backend.extraction.patterns import (
    ENTITY_PATTERNS,
    attribute_patterns,
    attribute_value_entities_patterns,
    relation_patterns,
    temporal_bounds,
)
from backend.extraction.sentences import split_sentences
from backend.ingestion.parser import location_of
from backend.ontology.schema import ATTRIBUTES, ENTITY_TYPES, RELATION_BY_NAME
from backend.schemas import DocumentExtraction, Fact, Mention, ParsedDocument

logger = logging.getLogger(__name__)

EXTRACTION_METHODS: dict[str, str] = {
    "regex": "Semantica 'regex' method + domain pattern pack (default, offline)",
    "pattern": "Semantica built-in 'pattern' method, unmodified (offline baseline)",
    "llm": "Semantica 'llm' method with the Groq provider (needs GROQ_API_KEY)",
}

_ATTRIBUTE_NAMES = {spec.name: spec for spec in ATTRIBUTES}
_LABEL_MAP = {
    "PERSON": "Person",
    "PER": "Person",
    "ORG": "Organization",
    "ORGANIZATION": "Organization",
    "PRODUCT": "Product",
    "TECHNOLOGY": "Technology",
    "TEAM": "Team",
    "PROJECT": "Project",
    "EVENT": "Event",
    "YEAR": "Year",
}
_TYPE_PRIORITY = {name: index for index, name in enumerate([*ENTITY_PATTERNS, "Year"])}


class ExtractionError(RuntimeError):
    pass


def make_fact_id(document_id: int, subject: str, predicate: str, obj: str, start: int) -> str:
    """Deterministic id: re-extracting the same statement yields the same id."""
    raw = f"{document_id}|{subject.lower()}|{predicate}|{obj.lower()}|{start}"
    return "fact_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


class ExtractionPipeline:
    def __init__(self, settings: Settings, method: str = "regex") -> None:
        if method not in EXTRACTION_METHODS:
            raise ExtractionError(f"Unknown extraction method '{method}'.")
        if method == "llm" and not settings.llm_available:
            raise ExtractionError("LLM extraction needs GROQ_API_KEY in the environment or .env.")
        self.method = method
        options: dict[str, Any] = {}
        if method == "llm":
            options = {
                "provider": "groq",
                "llm_model": settings.groq_model,
                "api_key": settings.groq_api_key.get_secret_value(),
            }
        self._ner = NERExtractor(method=method, **options)
        self._relations = RelationExtractor(method=method, **options)
        self._triplets = TripletExtractor(method="pattern")
        self._events = EventDetector(method="pattern")

    # ------------------------------------------------------------------ public
    def extract(self, document: ParsedDocument) -> DocumentExtraction:
        extracted_at = datetime.now(timezone.utc).isoformat()
        mentions: list[Mention] = []
        facts: dict[str, Fact] = {}
        discarded: Counter[str] = Counter()
        triplets: list[list[str]] = []

        for start, end, sentence in self._units(document):
            entities = self._extract_entities(sentence, discarded)
            for entity in entities:
                if entity["label"] == "Year":
                    continue
                absolute = start + entity["start"]
                mentions.append(
                    Mention(
                        text=entity["text"],
                        entity_type=entity["label"],
                        start=absolute,
                        end=start + entity["end"],
                        confidence=entity["confidence"],
                        method=self.method,
                        location=location_of(document, absolute, start + entity["end"]),
                    )
                )
            if len(entities) < 2:
                continue
            relations = self._extract_relations(sentence, entities, discarded)
            for relation, predicate, subject, obj in relations:
                fact = self._to_fact(
                    document, relation, predicate, subject, obj, sentence, start, end, extracted_at
                )
                facts.setdefault(fact.fact_id, fact)
            if relations:
                for triplet in self._triplets.extract_triplets(
                    sentence, relations=[item[0] for item in relations]
                ):
                    triplets.append([triplet.subject, triplet.predicate, triplet.object])

        return DocumentExtraction(
            document_id=document.document_id,
            filename=document.filename,
            method=self.method,
            extracted_at=extracted_at,
            mentions=mentions,
            facts=list(facts.values()),
            events=self._detect_events(document),
            triplets=triplets,
            discarded=dict(discarded),
        )

    # ----------------------------------------------------------------- helpers
    def _units(self, document: ParsedDocument) -> list[tuple[int, int, str]]:
        """Sentences for local methods; the LLM also works per sentence for provenance."""
        return split_sentences(document.text)

    def _extract_entities(self, sentence: str, discarded: Counter[str]) -> list[dict]:
        options: dict[str, Any] = {}
        if self.method == "regex":
            options["patterns"] = {**ENTITY_PATTERNS, **attribute_value_entities_patterns()}
        if self.method == "llm":
            options["entity_types"] = list(ENTITY_TYPES)
        candidates: list[dict] = []
        for entity in self._ner.extract_entities(sentence, **options):
            if self._is_library_fallback(entity.metadata):
                discarded["Semantica fallback entity (requested method found nothing)"] += 1
                continue
            label = _LABEL_MAP.get(str(entity.label).upper())
            if label is None:
                discarded[f"entity label {entity.label}"] += 1
                continue
            start, end = entity.start_char, entity.end_char
            if sentence[start:end] != entity.text:  # LLM offsets are not guaranteed
                start = sentence.find(entity.text)
                end = start + len(entity.text)
                if start < 0:
                    discarded["entity not found in text"] += 1
                    continue
            candidates.append(
                {
                    "text": entity.text,
                    "label": label,
                    "start": start,
                    "end": end,
                    "confidence": float(entity.confidence if entity.confidence is not None else 0.0),
                    "raw": entity,
                }
            )
        kept = resolve_overlaps(candidates)
        if overlapping := len(candidates) - len(kept):
            discarded["overlapping entity span"] += overlapping
        return kept

    def _extract_relations(
        self, sentence: str, entities: list[dict], discarded: Counter[str]
    ) -> list[tuple[Any, str, dict, dict]]:
        raw_entities = []
        for entity in entities:
            raw = entity["raw"]
            raw.start_char, raw.end_char = entity["start"], entity["end"]
            raw_entities.append(raw)

        options: dict[str, Any] = {}
        if self.method == "regex":
            texts = [entity["text"] for entity in entities]
            options["patterns"] = {**relation_patterns(texts), **attribute_patterns(texts)}
        if self.method == "llm":
            options["relation_types"] = [*RELATION_BY_NAME, *_ATTRIBUTE_NAMES]

        by_text = {entity["text"].lower(): entity for entity in entities}
        accepted: list[tuple[Any, str, dict, dict]] = []
        for relation in self._relations.extract_relations(sentence, raw_entities, **options):
            if self._is_library_fallback(relation.metadata):
                discarded["Semantica fallback relation (requested method found nothing)"] += 1
                continue
            predicate = str(relation.predicate).strip()
            name = predicate.upper().replace(" ", "_")
            if predicate.lower() in _ATTRIBUTE_NAMES:
                name = predicate.lower()
            if name not in RELATION_BY_NAME and name not in _ATTRIBUTE_NAMES:
                # e.g. Semantica's generic fallback: founded_by, related_to
                discarded[f"predicate outside ontology: {predicate}"] += 1
                continue
            subject = by_text.get(relation.subject.text.lower())
            obj = by_text.get(relation.object.text.lower())
            if subject is None or obj is None or subject is obj:
                discarded["relation endpoint not extracted"] += 1
                continue
            if not _types_allowed(name, subject["label"], obj["label"]):
                discarded[f"domain/range violation: {name}"] += 1
                continue
            accepted.append((relation, name, subject, obj))
        return accepted

    def _is_library_fallback(self, metadata: dict | None) -> bool:
        """Detect results Semantica produced by silently switching methods.

        When the requested method returns nothing, Semantica 0.7.0 falls back to
        generic patterns, and finally to "capitalised word" entities or
        adjacent-pair ``related_to`` relations.  Those must not enter the graph.
        """
        used = str((metadata or {}).get("extraction_method", ""))
        if used.startswith("last_resort"):
            return True
        return self.method != "pattern" and used == "pattern"

    def _to_fact(
        self,
        document: ParsedDocument,
        relation: Any,
        predicate: str,
        subject: dict,
        obj: dict,
        sentence: str,
        start: int,
        end: int,
        extracted_at: str,
    ) -> Fact:
        is_attribute = predicate in _ATTRIBUTE_NAMES
        valid_from, valid_until = temporal_bounds(sentence)
        return Fact(
            fact_id=make_fact_id(
                document.document_id, subject["text"], predicate, obj["text"], start
            ),
            kind="attribute" if is_attribute else "relation",
            subject=subject["text"],
            subject_type=subject["label"],
            predicate=predicate,
            object=obj["text"],
            object_type="Literal" if is_attribute else obj["label"],
            quote=sentence,
            location=location_of(document, start, end),
            start=start,
            end=end,
            confidence=float(relation.confidence),
            method=self.method,
            extracted_at=extracted_at,
            valid_from=valid_from,
            valid_until=valid_until,
        )

    def _detect_events(self, document: ParsedDocument) -> list[dict]:
        """Exploratory: Semantica's pattern EventDetector (shown in the UI, not graphed)."""
        try:
            events = self._events.detect_events(document.text)
        except Exception as exc:  # exploratory feature must never fail ingestion
            logger.warning("Event detection failed for %s: %s", document.filename, exc)
            return []
        return [
            {
                "text": event.text,
                "event_type": event.event_type,
                "participants": list(event.participants),
                "time": event.time,
                "location": location_of(document, event.start_char, event.end_char),
            }
            for event in events
        ]


def resolve_overlaps(candidates: list[dict]) -> list[dict]:
    """Keep the longest span; on ties use the pattern priority order."""
    ordered = sorted(
        candidates,
        key=lambda c: (-(c["end"] - c["start"]), _TYPE_PRIORITY.get(c["label"], 99), c["start"]),
    )
    kept: list[dict] = []
    for candidate in ordered:
        if all(candidate["end"] <= k["start"] or candidate["start"] >= k["end"] for k in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda c: c["start"])


def _types_allowed(predicate: str, subject_type: str, object_type: str) -> bool:
    if predicate in _ATTRIBUTE_NAMES:
        return subject_type in _ATTRIBUTE_NAMES[predicate].subject_types
    spec = RELATION_BY_NAME[predicate]
    return subject_type in spec.domain and object_type in spec.range
