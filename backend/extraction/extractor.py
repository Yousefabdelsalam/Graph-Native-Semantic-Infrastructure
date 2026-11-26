"""Entity, relationship, triplet and event extraction through Semantica.

Division of labour
------------------
Semantica:    ``NERExtractor``, ``RelationExtractor``, ``extract_entities_llm`` /
              ``extract_relations_llm``, ``TripletExtractor`` and ``EventDetector``
              do all matching / model calls.
Application:  supplies the domain patterns, chooses the text unit (a sentence for
              rules; a passage with re-joined PDF lines for the model), locates the
              supporting statement, enforces the ontology's domain/range, and
              records what was discarded and why.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable

from semantica.semantic_extract import (
    EventDetector,
    NERExtractor,
    RelationExtractor,
    TripletExtractor,
)
from semantica.semantic_extract.methods import extract_entities_llm, extract_relations_llm

from backend.config import Settings
from backend.extraction.patterns import (
    ENTITY_PATTERNS,
    attribute_patterns,
    attribute_value_entities_patterns,
    relation_patterns,
    temporal_bounds,
)
from backend.extraction.sentences import split_sentences, unwrap_lines
from backend.ingestion.parser import location_of
from backend.llm.provider import register as register_llm_provider
from backend.ontology.schema import (
    ALL_ENTITY_TYPES,
    ALL_RELATION_BY_NAME,
    ATTRIBUTES,
    RELATION_BY_NAME,
    llm_endpoints,
)
from backend.schemas import DocumentExtraction, Fact, Mention, ParsedDocument

logger = logging.getLogger(__name__)

EXTRACTION_METHODS: dict[str, str] = {
    "regex": "Semantica 'regex' method + domain pattern pack (default, offline)",
    "pattern": "Semantica built-in 'pattern' method, unmodified (offline baseline)",
    "llm": "AI extraction with OpenAI through Semantica (needs OPENAI_API_KEY)",
}

# The thresholds Semantica's extractor classes apply by default.
MIN_ENTITY_CONFIDENCE = 0.5
MIN_RELATION_CONFIDENCE = 0.6
# Size of the passage sent to the model, and the longest quote stored as evidence
# when two names never share a sentence.
PASSAGE_CHARS = 1800
MAX_EVIDENCE_CHARS = 500

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
    "CONCEPT": "Concept",
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
            raise ExtractionError("AI extraction needs OPENAI_API_KEY in the environment or .env.")
        self.method = method
        self._llm_options: dict[str, Any] = {}
        if method == "llm":
            # Semantica's NERExtractor / RelationExtractor catch provider errors and
            # fall back to other methods, so the LLM functions are called directly.
            self._llm_options = {
                "provider": register_llm_provider(),
                "model": settings.openai_model,
                "api_key": settings.openai_api_key.get_secret_value(),
                "reasoning_effort": settings.openai_reasoning_effort,
                "silent_fail": False,
            }
        else:
            self._ner = NERExtractor(method=method)
            self._relations = RelationExtractor(method=method)
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
                if self.method == "llm":
                    quote, low, high = locate_evidence(sentence, subject["text"], obj["text"])
                    span = (quote, start + low, start + high)
                else:
                    span = (sentence, start, end)
                fact = self._to_fact(
                    document, relation, predicate, subject, obj, *span, extracted_at
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
        """Sentences for the rule methods; passages for the model.

        Offsets always refer to ``document.text``.
        """
        if self.method != "llm":
            return split_sentences(document.text)
        return passages(document)

    def _raw_entities(self, sentence: str) -> list[Any]:
        if self.method == "llm":
            found = self._call_llm(
                extract_entities_llm, sentence, entity_types=list(ALL_ENTITY_TYPES)
            )
            return [e for e in found if (e.confidence or 0.0) >= MIN_ENTITY_CONFIDENCE]
        options: dict[str, Any] = {}
        if self.method == "regex":
            options["patterns"] = {**ENTITY_PATTERNS, **attribute_value_entities_patterns()}
        return self._ner.extract_entities(sentence, **options)

    def _raw_relations(self, sentence: str, entities: list[dict], raw: list[Any]) -> list[Any]:
        if self.method == "llm":
            found = self._call_llm(
                extract_relations_llm,
                sentence,
                raw,
                relation_types=[*ALL_RELATION_BY_NAME, *_ATTRIBUTE_NAMES],
            )
            return [r for r in found if (r.confidence or 0.0) >= MIN_RELATION_CONFIDENCE]
        options: dict[str, Any] = {}
        if self.method == "regex":
            texts = [entity["text"] for entity in entities]
            options["patterns"] = {**relation_patterns(texts), **attribute_patterns(texts)}
        return self._relations.extract_relations(sentence, raw, **options)

    def _call_llm(self, function: Callable[..., list], *args: Any, **options: Any) -> list:
        """Run a Semantica LLM extraction; any failure aborts the whole document.

        Partial results are never returned: ingesting them would remove the
        document's earlier facts from the graph.
        """
        try:
            return function(*args, **self._llm_options, **options)
        except Exception as exc:
            key = self._llm_options["api_key"]
            reason = str(exc).replace(key, "***")[:300]
            raise ExtractionError(f"LLM extraction failed: {reason}") from exc

    def _extract_entities(self, sentence: str, discarded: Counter[str]) -> list[dict]:
        candidates: list[dict] = []
        for entity in self._raw_entities(sentence):
            if self._is_library_fallback(entity.metadata):
                discarded["Semantica fallback entity (requested method found nothing)"] += 1
                continue
            label = _LABEL_MAP.get(str(entity.label).upper())
            if label is None:
                discarded[f"entity label {entity.label}"] += 1
                continue
            start, end = entity.start_char, entity.end_char
            text = entity.text
            if sentence[start:end] != text:  # the model returns no usable offsets
                found = find_name(sentence, text)
                if found is None:
                    discarded["entity not found in text"] += 1
                    continue
                start, end = found
                if self.method == "llm":
                    text = sentence[start:end]  # use the document's own spelling
                    entity.text = text
            candidates.append(
                {
                    "text": text,
                    "label": label,
                    "start": start,
                    "end": end,
                    "confidence": float(entity.confidence if entity.confidence is not None else 0.0),
                    "raw": entity,
                }
            )
        if self.method == "llm":
            # Nested names ("agent" inside "AI agent") are distinct entities here;
            # only repeats of the same name are dropped.
            unique: dict[str, dict] = {}
            for candidate in candidates:
                unique.setdefault(candidate["text"].lower(), candidate)
            return sorted(unique.values(), key=lambda c: c["start"])
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

        by_text = {entity["text"].lower(): entity for entity in entities}
        accepted: list[tuple[Any, str, dict, dict]] = []
        for relation in self._raw_relations(sentence, entities, raw_entities):
            if self._is_library_fallback(relation.metadata):
                discarded["Semantica fallback relation (requested method found nothing)"] += 1
                continue
            predicate = str(relation.predicate).strip()
            name = predicate.upper().replace(" ", "_")
            if predicate.lower() in _ATTRIBUTE_NAMES:
                name = predicate.lower()
            known = ALL_RELATION_BY_NAME if self.method == "llm" else RELATION_BY_NAME
            if name not in known and name not in _ATTRIBUTE_NAMES:
                # e.g. Semantica's generic fallback: founded_by, related_to
                discarded[f"predicate outside ontology: {predicate}"] += 1
                continue
            subject = by_text.get(relation.subject.text.lower())
            obj = by_text.get(relation.object.text.lower())
            if subject is None or obj is None or subject is obj:
                discarded["relation endpoint not extracted"] += 1
                continue
            if not _types_allowed(name, subject["label"], obj["label"], self.method):
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


def _types_allowed(predicate: str, subject_type: str, object_type: str, method: str) -> bool:
    if predicate in _ATTRIBUTE_NAMES:
        return subject_type in _ATTRIBUTE_NAMES[predicate].subject_types
    if method == "llm":
        domain, allowed = llm_endpoints(predicate)
    else:
        spec = RELATION_BY_NAME[predicate]
        domain, allowed = spec.domain, spec.range
    return subject_type in domain and object_type in allowed


def passages(document: ParsedDocument, size: int = PASSAGE_CHARS) -> list[tuple[int, int, str]]:
    """Whole sentences of one page or section, grouped up to ``size`` characters."""
    result: list[tuple[int, int, str]] = []
    for segment in document.segments:
        body = unwrap_lines(document.text[segment.start : segment.end])
        current: list[tuple[int, int]] = []

        def flush() -> None:
            if current:
                low, high = current[0][0], current[-1][1]
                result.append((segment.start + low, segment.start + high, body[low:high]))
                current.clear()

        for low, high, _ in split_sentences(body):
            if current and high - current[0][0] > size:
                flush()
            current.append((low, high))
        flush()
    return result


def find_name(text: str, name: str) -> tuple[int, int] | None:
    """First whole-word occurrence of ``name``, ignoring case and line breaks."""
    words = [re.escape(word) for word in name.split()]
    if not words:
        return None
    pattern = r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.span() if match else None


def locate_evidence(passage: str, subject: str, obj: str) -> tuple[str, int, int]:
    """The statement supporting a model-extracted relation: ``(quote, start, end)``.

    Prefers one sentence naming both ends.  Otherwise the stretch from the first
    name to the other (marked by "…" when it has to be shortened).
    """
    sentences = split_sentences(passage)
    for low, high, sentence in sentences:
        if find_name(sentence, subject) and find_name(sentence, obj):
            return " ".join(sentence.split()), low, high
    spans = [span for span in (find_name(passage, subject), find_name(passage, obj)) if span]
    first = min(span[0] for span in spans)
    last = max(span[1] for span in spans)
    low = next((s for s, e, _ in sentences if s <= first < e), first)
    high = next((e for s, e, _ in sentences if s < last <= e), last)
    quote = " ".join(passage[low:high].split())
    if len(quote) > MAX_EVIDENCE_CHARS:
        half = MAX_EVIDENCE_CHARS // 2
        quote = f"{quote[:half]} … {quote[-half:]}"
    return quote, low, high
