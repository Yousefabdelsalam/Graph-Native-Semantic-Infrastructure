"""Conflict detection over single-valued claims.

Semantica:    ``ConflictDetector.detect_value_conflicts`` finds disagreements,
              ``SourceTracker`` records source credibility and
              ``ConflictResolver`` *suggests* a winner.
Application:  separates genuine contradictions from claims whose validity
              periods do not overlap (a succession, not a conflict), and never
              applies a suggested resolution to the graph.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import isclose

from semantica.conflicts import ConflictDetector, ConflictResolver, SourceTracker

DEFAULT_CREDIBILITY = 0.5


def _overlaps(first: dict, second: dict) -> bool:
    """Open-ended bounds are treated as unlimited; ISO dates compare as text."""
    start = max(first.get("valid_from") or "0000-00-00", second.get("valid_from") or "0000-00-00")
    end = min(first.get("valid_until") or "9999-12-31", second.get("valid_until") or "9999-12-31")
    return start <= end


def _claim(observation: dict) -> dict:
    keys = (
        "value",
        "quote",
        "confidence",
        "fact_id",
        "document_id",
        "valid_from",
        "valid_until",
    )
    claim = {key: observation.get(key) for key in keys}
    claim["document"] = observation["source"]
    claim["location"] = observation["section"]
    return claim


def _is_tie(strategy: str, claims: list[dict], credibility: dict[str, float]) -> bool:
    """Whether the evidence does not separate the top values.

    ``ConflictResolver`` still returns a value then, so it must be labelled as
    an arbitrary pick.
    """
    weights: dict[str, float] = defaultdict(float)
    for claim in claims:
        weight = 1.0
        if strategy == "credibility_weighted":
            weight = claim["confidence"] * credibility.get(claim["source"], DEFAULT_CREDIBILITY)
        weights[claim["value"]] += weight
    top = max(weights.values())
    return sum(isclose(w, top) for w in weights.values()) > 1


def detect_conflicts(
    observations: list[dict], credibility: dict[str, float] | None = None
) -> tuple[list[dict], list[dict]]:
    """Return ``(conflicts, successions)``."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for observation in observations:
        grouped[(observation["id"], observation["property"])].append(observation)

    tracker = SourceTracker()
    for document in {o["source"] for o in observations}:
        tracker.set_source_credibility(
            document, (credibility or {}).get(document, DEFAULT_CREDIBILITY)
        )
    detector = ConflictDetector(source_tracker=tracker)
    resolver = ConflictResolver(source_tracker=tracker)

    conflicts: list[dict] = []
    successions: list[dict] = []
    for (entity_id, property_name), claims in sorted(grouped.items()):
        if len({c["value"] for c in claims}) < 2:
            continue
        clashing: set[int] = set()
        for (i, first), (j, second) in combinations(enumerate(claims), 2):
            if first["value"] != second["value"] and _overlaps(first, second):
                clashing.update((i, j))
        if not clashing:
            successions.append(
                {
                    "entity_id": entity_id,
                    "entity_name": claims[0]["name"],
                    "property": property_name,
                    "claims": [
                        _claim(c) for c in sorted(claims, key=lambda c: c.get("valid_from") or "")
                    ],
                }
            )
            continue

        disputed = [claims[i] for i in sorted(clashing)]
        for conflict in detector.detect_value_conflicts(disputed, property_name):
            suggestions = []
            for strategy in ("voting", "credibility_weighted"):
                outcome = resolver.resolve_conflict(conflict, strategy=strategy)
                suggestions.append(
                    {
                        "strategy": strategy,
                        "value": outcome.resolved_value,
                        "confidence": round(float(outcome.confidence), 3),
                        "notes": outcome.resolution_notes,
                        "tie": _is_tie(strategy, disputed, credibility or {}),
                    }
                )
            conflicts.append(
                {
                    "conflict_id": conflict.conflict_id,
                    "conflict_type": conflict.conflict_type.value,
                    "entity_id": entity_id,
                    "entity_name": disputed[0]["name"],
                    "property": property_name,
                    "values": sorted({str(v) for v in conflict.conflicting_values}),
                    "severity": conflict.severity,
                    "recommended_action": conflict.recommended_action,
                    "claims": [_claim(c) for c in disputed],
                    "suggestions": suggestions,
                }
            )
    return conflicts, successions
