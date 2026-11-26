"""Entity resolution: which surface forms refer to the same real-world entity?

Three layers, from most to least certain; every merge records which layer made it.

1. **Normalised name** (Semantica ``EntityNormalizer`` + the curated alias
   table; application strips honorifics and legal suffixes first).
2. **Spelling variants** (application rules, ``Technology`` / ``Concept`` /
   ``Product`` only): hyphen or space, "Long Name (ACR)" with the matching
   acronym, and a plural when the singular is also attested.
3. **Unique partial person name** (application rule): "Sarah" is merged into
   "Sarah Chen" only when exactly one known full name starts with "Sarah".
4. **Similarity** (Semantica ``DuplicateDetector``): pairs above
   ``auto_merge_threshold`` are merged, the rest are listed for human review.

Merged records are combined with Semantica ``EntityMerger``.  The only types
merged with each other are ``Technology``, ``Concept`` and ``Product``, which
extractors assign interchangeably to the same name; people, organisations,
teams, projects and events never merge across types.  Every original surface
form is kept as an alias, and every observed type and source is recorded.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from semantica.deduplication import DuplicateDetector, EntityMerger
from semantica.normalize import EntityNormalizer

from backend.graph.vocabulary import ALIASES, HONORIFICS, LEGAL_SUFFIXES
from backend.schemas import DocumentExtraction

logger = logging.getLogger(__name__)

# Types that may describe one thing, in order of preference when mentions tie.
INTERCHANGEABLE_TYPES = ("Technology", "Concept", "Product")
_INTERCHANGEABLE = "/".join(INTERCHANGEABLE_TYPES)
_ACRONYM_SUFFIX = re.compile(r"^(.*\S)\s*\(([A-Za-z][A-Za-z0-9\-]{1,7})\)$")
_DASHES = re.compile(r"[\s\-\u2010-\u2015_]+")
_NOTE_ORDER = ("acronym", "plural form", "across types")


@dataclass
class ResolvedEntities:
    entities: dict[str, dict]  # canonical id -> entity record
    lookup: dict[tuple[str, str], str]  # (type, surface form lower) -> canonical id
    merges: list[dict] = field(default_factory=list)
    review: list[dict] = field(default_factory=list)

    def resolve(self, entity_type: str, surface: str) -> str | None:
        found = self.lookup.get((entity_type, surface.lower()))
        if found is None and entity_type in INTERCHANGEABLE_TYPES:
            for other in INTERCHANGEABLE_TYPES:
                found = found or self.lookup.get((other, surface.lower()))
        return found


def type_group(entity_type: str) -> str:
    return _INTERCHANGEABLE if entity_type in INTERCHANGEABLE_TYPES else entity_type


def spelling_key(name: str) -> str:
    """Lower case, hyphens as spaces, without a matching "(ACR)" suffix."""
    if match := _ACRONYM_SUFFIX.match(name):
        initials = "".join(w[0] for w in _DASHES.split(match.group(1)) if w)
        if initials.lower() == match.group(2).lower():
            name = match.group(1)
    return _DASHES.sub(" ", name).strip().lower()


def singular_key(key: str, surfaces: list[str]) -> str:
    """The key with its last word in the singular (a candidate, not a claim)."""
    word = key.split(" ")[-1]
    head = key[: len(key) - len(word)]
    if any(re.search(r"[A-Z]{2}s$", s) for s in surfaces):  # "APIs", "LLMs"
        return head + word[:-1]
    if len(word) <= 3:
        return key
    if word.endswith("ies"):
        return head + word[:-3] + "y"
    if word.endswith(("sses", "xes", "ches", "shes")):
        return head + word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return head + word[:-1]
    return key


def slug(entity_type: str, name: str) -> str:
    body = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{entity_type.lower()}:{body}"


def clean_surface(name: str) -> str:
    """Remove honorifics and legal suffixes ("Dr. X", "X Inc.")."""
    words = " ".join(name.split()).split(" ")
    if len(words) > 1 and words[0].lower() in HONORIFICS:
        words = words[1:]
    if len(words) > 1 and words[-1].lower() in LEGAL_SUFFIXES:
        words = words[:-1]
    return " ".join(words).rstrip(",")


class EntityResolution:
    def __init__(
        self,
        auto_merge_threshold: float = 0.93,
        review_threshold: float = 0.60,
    ) -> None:
        self.auto_merge_threshold = auto_merge_threshold
        self.review_threshold = review_threshold
        self._normalizer = EntityNormalizer(alias_map=ALIASES)
        self._detector = DuplicateDetector(
            similarity_threshold=review_threshold, confidence_threshold=review_threshold
        )
        self._merger = EntityMerger()

    def normalized_name(self, surface: str, entity_type: str) -> str:
        cleaned = clean_surface(surface)
        return self._normalizer.resolve_aliases(cleaned, entity_type=entity_type) or cleaned

    # ------------------------------------------------------------------ public
    def resolve(self, extractions: list[DocumentExtraction]) -> ResolvedEntities:
        records = self._collect(extractions)
        merges: list[dict] = []

        # Layers 1 and 2: normalised name and spelling variants within a type group.
        clusters: dict[tuple[str, str], list[dict]] = defaultdict(list)
        notes: dict[tuple[str, str], set[str]] = defaultdict(set)
        for record in records:
            canonical = self.normalized_name(record["surface"], record["type"])
            group = type_group(record["type"])
            name = spelling_key(canonical) if group == _INTERCHANGEABLE else canonical.lower()
            clusters[(group, name)].append(record)
        self._merge_acronyms(clusters, notes)
        self._merge_plurals(clusters, notes)

        for key, members in clusters.items():
            surfaces = sorted({m["surface"] for m in members})
            types = sorted({m["type"] for m in members})
            if len(surfaces) > 1 or len(types) > 1:
                if len(types) > 1:
                    notes[key].add("across types")
                reasons = [n for n in _NOTE_ORDER if n in notes[key]]
                if any(
                    self.normalized_name(m["surface"], m["type"]) != clean_surface(m["surface"])
                    for m in members
                ):
                    reasons.insert(0, "curated alias")
                entity_type = self._primary_type(members)
                merges.append(
                    {
                        "type": entity_type,
                        "canonical": self._display_name(entity_type, members),
                        "merged": surfaces,
                        "observed_types": types,
                        "method": ", ".join(["normalised name", *reasons])
                        if reasons
                        else "normalised name / curated alias",
                        "score": 1.0,
                    }
                )

        merges += self._merge_partial_person_names(clusters)
        similarity_merges, review = self._merge_similar(clusters)
        merges += similarity_merges

        entities: dict[str, dict] = {}
        lookup: dict[tuple[str, str], str] = {}
        for _, members in sorted(clusters.items()):
            entity = self._build_entity(self._primary_type(members), members)
            entities[entity["id"]] = entity
            for member in members:
                lookup[(member["type"], member["surface"].lower())] = entity["id"]
            for alias in entity["aliases"]:
                for observed in entity["observed_types"]:
                    lookup.setdefault((observed, alias.lower()), entity["id"])
        return ResolvedEntities(entities, lookup, merges, review)

    # ----------------------------------------------------------------- layers
    def _merge_acronyms(self, clusters: dict, notes: dict) -> None:
        """Join a bare "CoT" to "Chain of Thought (CoT)" when only one expansion exists."""
        expansions: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for key, members in clusters.items():
            for member in members:
                match = _ACRONYM_SUFFIX.match(member["surface"])
                if key[0] == _INTERCHANGEABLE and match:
                    if spelling_key(member["surface"]) == spelling_key(match.group(1)):
                        expansions[match.group(2)].add(key)
        for acronym, targets in sorted(expansions.items()):
            source = (_INTERCHANGEABLE, spelling_key(acronym))
            members = clusters.get(source, [])
            if len(targets) != 1 or not members or source in targets or not targets <= set(clusters):
                continue
            if all(m["surface"] == acronym for m in members):
                target = next(iter(targets))
                clusters[target].extend(clusters.pop(source))
                notes[target].add("acronym")

    def _merge_plurals(self, clusters: dict, notes: dict) -> None:
        """Join a plural to its singular, only when both are attested."""
        for key in sorted(k for k in clusters if k[0] == _INTERCHANGEABLE):
            surfaces = [m["surface"] for m in clusters[key]]
            target = (key[0], singular_key(key[1], surfaces))
            if target != key and target in clusters:
                clusters[target].extend(clusters.pop(key))
                notes[target] |= notes.pop(key, set()) | {"plural form"}
    def _merge_partial_person_names(
        self, clusters: dict[tuple[str, str], list[dict]]
    ) -> list[dict]:
        merges: list[dict] = []
        full = [key for key in clusters if key[0] == "Person" and " " in key[1]]
        for key in [k for k in clusters if k[0] == "Person" and " " not in k[1]]:
            matches = [other for other in full if other[1].split()[0] == key[1]]
            if len(matches) != 1:
                continue
            target = matches[0]
            partial = clusters.pop(key)
            surfaces = sorted({m["surface"] for m in partial})
            clusters[target].extend(partial)
            merges.append(
                {
                    "type": "Person",
                    "canonical": self._display_name("Person", clusters[target]),
                    "merged": surfaces,
                    "method": "unique first-name match (application rule)",
                    "score": 0.8,
                }
            )
        return merges

    def _merge_similar(
        self, clusters: dict[tuple[str, str], list[dict]]
    ) -> tuple[list[dict], list[dict]]:
        by_type: dict[str, list[dict]] = defaultdict(list)
        group_of: dict[tuple[str, str], str] = {}
        for key, members in clusters.items():
            entity_type = self._primary_type(members)
            name = self._display_name(entity_type, members)
            by_type[entity_type].append({"id": key[1], "name": name, "type": entity_type})
            group_of[(entity_type, key[1])] = key[0]

        merges: list[dict] = []
        review: list[dict] = []
        parents: dict[tuple[str, str], tuple[str, str]] = {}

        def root(key: tuple[str, str]) -> tuple[str, str]:
            while key in parents:
                key = parents[key]
            return key

        for entity_type, items in by_type.items():
            if len(items) < 2:
                continue
            for candidate in self._detector.detect_duplicates(items):
                first, second = candidate.entity1, candidate.entity2
                row = {
                    "type": entity_type,
                    "first": first["name"],
                    "second": second["name"],
                    "similarity": round(float(candidate.similarity_score), 3),
                    "confidence": round(float(candidate.confidence), 3),
                    "reasons": list(candidate.reasons),
                }
                if candidate.similarity_score >= self.auto_merge_threshold:
                    a = root((group_of[(entity_type, first["id"])], first["id"]))
                    b = root((group_of[(entity_type, second["id"])], second["id"]))
                    if a == b:
                        continue
                    clusters[a].extend(clusters.pop(b))
                    parents[b] = a
                    merges.append(
                        {
                            "type": entity_type,
                            "canonical": first["name"],
                            "merged": [first["name"], second["name"]],
                            "method": "Semantica DuplicateDetector similarity",
                            "score": row["similarity"],
                        }
                    )
                else:
                    review.append(row)
        review.sort(key=lambda r: r["similarity"], reverse=True)
        return merges, review

    # ----------------------------------------------------------------- helpers
    def _collect(self, extractions: list[DocumentExtraction]) -> list[dict]:
        """One record per (type, surface form, document)."""
        counts: Counter[tuple[str, str, int]] = Counter()
        confidence: dict[tuple[str, str, int], float] = {}
        by_model: set[tuple[str, str, int]] = set()
        filenames: dict[int, str] = {}
        for extraction in extractions:
            filenames[extraction.document_id] = extraction.filename
            for mention in extraction.mentions:
                key = (mention.entity_type, mention.text, extraction.document_id)
                counts[key] += 1
                confidence[key] = max(confidence.get(key, 0.0), mention.confidence)
                if mention.method == "llm":
                    by_model.add(key)
        return [
            {
                "type": entity_type,
                "surface": surface,
                "document_id": document_id,
                "filename": filenames[document_id],
                "mentions": count,
                "confidence": confidence[(entity_type, surface, document_id)],
                "by_model": (entity_type, surface, document_id) in by_model,
            }
            for (entity_type, surface, document_id), count in sorted(counts.items())
        ]

    def _primary_type(self, members: list[dict]) -> str:
        """The type an AI extraction assigned if any, else the pattern one; by mentions."""
        preferred = [m for m in members if m["by_model"]] or members
        weights: Counter[str] = Counter()
        for member in preferred:
            weights[member["type"]] += member["mentions"]

        def rank(entity_type: str) -> tuple[int, int, str]:
            order = (
                INTERCHANGEABLE_TYPES.index(entity_type)
                if entity_type in INTERCHANGEABLE_TYPES
                else 0
            )
            return -weights[entity_type], order, entity_type

        return min(weights, key=rank)

    def _display_name(self, entity_type: str, members: list[dict]) -> str:
        """Most frequent normalised spelling; prefer the fullest person name.

        A spelling in capitals (usually a heading) or in the plural gives way to
        an ordinary one.
        """
        weights: Counter[str] = Counter()
        for member in members:
            weights[self.normalized_name(member["surface"], entity_type)] += member["mentions"]
        if entity_type == "Person":
            return max(weights, key=lambda n: (len(n.split()), weights[n], n))
        keys = {spelling_key(name) for name in weights}

        def rank(name: str) -> tuple[bool, bool, int, int, str]:
            shouting = name.isupper() and (" " in name or len(name) > 6)
            key = spelling_key(name)
            singular = singular_key(key, [name])
            plural = singular != key and singular in keys
            return (not shouting, not plural, weights[name], len(name), name)

        return max(weights, key=rank)

    def _build_entity(self, entity_type: str, members: list[dict]) -> dict:
        name = self._display_name(entity_type, members)
        entity_id = slug(entity_type, name)
        group = [
            {
                "id": entity_id,
                "name": name,
                "type": entity_type,
                "properties": {"document_id": member["document_id"]},
                "confidence": member["confidence"],
            }
            for member in members
        ]
        confidence = max(member["confidence"] for member in members)
        if len(group) > 1:
            merged = self._merger.merge_entity_group(group, strategy="keep_most_complete")
            confidence = max(confidence, float(merged.merged_entity.get("confidence", 0.0)))
        aliases = sorted({member["surface"] for member in members} - {name})
        return {
            "id": entity_id,
            "name": name,
            "type": entity_type,
            "observed_types": sorted({member["type"] for member in members}),
            "normalized_name": name.lower(),
            "aliases": aliases,
            "source_documents": sorted({member["filename"] for member in members}),
            "document_ids": sorted({member["document_id"] for member in members}),
            "mention_count": sum(member["mentions"] for member in members),
            "confidence": round(confidence, 3),
        }


def pairwise_review_note(entities: dict[str, dict]) -> list[tuple[str, str]]:
    """Cross-type name collisions, e.g. a Team and a Product sharing a name."""
    return [
        (a["id"], b["id"])
        for a, b in combinations(entities.values(), 2)
        if a["normalized_name"] == b["normalized_name"] and a["type"] != b["type"]
    ]
