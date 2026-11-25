"""Entity resolution: which surface forms refer to the same real-world entity?

Three layers, from most to least certain; every merge records which layer made it.

1. **Normalised name** (Semantica ``EntityNormalizer`` + the curated alias
   table; application strips honorifics and legal suffixes first).
2. **Unique partial person name** (application rule): "Sarah" is merged into
   "Sarah Chen" only when exactly one known full name starts with "Sarah".
3. **Similarity** (Semantica ``DuplicateDetector``): pairs above
   ``auto_merge_threshold`` are merged, the rest are listed for human review.

Merged records are combined with Semantica ``EntityMerger``.  Nothing is merged
across entity types, and every original surface form is kept as an alias.
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


@dataclass
class ResolvedEntities:
    entities: dict[str, dict]  # canonical id -> entity record
    lookup: dict[tuple[str, str], str]  # (type, surface form lower) -> canonical id
    merges: list[dict] = field(default_factory=list)
    review: list[dict] = field(default_factory=list)

    def resolve(self, entity_type: str, surface: str) -> str | None:
        return self.lookup.get((entity_type, surface.lower()))


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

        # Layer 1: identical normalised name within a type.
        clusters: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for record in records:
            canonical = self.normalized_name(record["surface"], record["type"])
            clusters[(record["type"], canonical.lower())].append(record)
        for (entity_type, _), members in clusters.items():
            surfaces = sorted({m["surface"] for m in members})
            if len(surfaces) > 1:
                merges.append(
                    {
                        "type": entity_type,
                        "canonical": self._display_name(entity_type, members),
                        "merged": surfaces,
                        "method": "normalised name / curated alias",
                        "score": 1.0,
                    }
                )

        merges += self._merge_partial_person_names(clusters)
        similarity_merges, review = self._merge_similar(clusters)
        merges += similarity_merges

        entities: dict[str, dict] = {}
        lookup: dict[tuple[str, str], str] = {}
        for (entity_type, _), members in sorted(clusters.items()):
            entity = self._build_entity(entity_type, members)
            entities[entity["id"]] = entity
            for member in members:
                lookup[(entity_type, member["surface"].lower())] = entity["id"]
            for alias in entity["aliases"]:
                lookup.setdefault((entity_type, alias.lower()), entity["id"])
        return ResolvedEntities(entities, lookup, merges, review)

    # ----------------------------------------------------------------- layers
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
        for key, members in clusters.items():
            name = self._display_name(key[0], members)
            by_type[key[0]].append({"id": key[1], "name": name, "type": key[0]})

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
                    a = root((entity_type, first["id"]))
                    b = root((entity_type, second["id"]))
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
        filenames: dict[int, str] = {}
        for extraction in extractions:
            filenames[extraction.document_id] = extraction.filename
            for mention in extraction.mentions:
                key = (mention.entity_type, mention.text, extraction.document_id)
                counts[key] += 1
                confidence[key] = max(confidence.get(key, 0.0), mention.confidence)
        return [
            {
                "type": entity_type,
                "surface": surface,
                "document_id": document_id,
                "filename": filenames[document_id],
                "mentions": count,
                "confidence": confidence[(entity_type, surface, document_id)],
            }
            for (entity_type, surface, document_id), count in sorted(counts.items())
        ]

    def _display_name(self, entity_type: str, members: list[dict]) -> str:
        """Most frequent normalised spelling; prefer the fullest person name."""
        weights: Counter[str] = Counter()
        for member in members:
            weights[self.normalized_name(member["surface"], entity_type)] += member["mentions"]
        if entity_type == "Person":
            return max(weights, key=lambda n: (len(n.split()), weights[n], n))
        return max(weights, key=lambda n: (weights[n], len(n), n))

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
