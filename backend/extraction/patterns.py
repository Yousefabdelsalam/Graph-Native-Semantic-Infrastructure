"""Domain pattern pack for Semantica's native ``regex`` extraction methods.

Semantica 0.7.0's ``pattern`` method only knows PERSON / ORG / GPE / DATE and
five relation types (founded_by, located_in, works_for, born_in, acquired_by).
Its ``regex`` method accepts caller-supplied patterns, which is the supported
extension point used here: **Semantica runs the matching, the application only
supplies the patterns.**

Two Semantica behaviours shape these patterns:

* entity regexes are applied with ``re.IGNORECASE``, so case-sensitive parts are
  wrapped in ``(?-i:...)``;
* relation regexes must expose ``subject`` / ``object`` groups whose text equals
  an extracted entity, so they are generated per document from those entities.
"""

from __future__ import annotations

import re
from typing import Iterable

from backend.ontology.schema import ATTRIBUTES, RELATIONS

# Technologies are a closed vocabulary: a gazetteer is the usual rule-based
# approach, and capitalisation alone cannot separate "PostgreSQL" from a product.
TECHNOLOGY_GAZETTEER: tuple[str, ...] = (
    "PostgreSQL",
    "Postgres",
    "MongoDB",
    "MySQL",
    "Redis",
    "Kafka",
    "PyTorch",
    "TensorFlow",
    "ONNX Runtime",
    "FastAPI",
    "Kubernetes",
    "Docker",
    "Neo4j",
)

_WORD = r"[A-Z][A-Za-z0-9]+"
# "Sarah Chen" but not "Project Atlas" / "Platform Team" / "Applied Vision".
_NOT_A_NAME = (
    r"(?!(?:The|Project|Platform|Team|Acme|Applied|Her|His|Their|This|An?|Archived|"
    r"Document|Vision|Product|Company|Engineering|Technology|Datastores|Caching|Employees|Teams)\b)"
)
_PERSON_VERBS = (
    r"(?:works|leads|led|has\s+led|heads|manages|collaborates|is\s+a\s+member|partners)"
)

# Order matters: when two matches overlap, the earlier type wins (see
# ``resolve_overlaps`` in extractor.py).
ENTITY_PATTERNS: dict[str, str] = {
    "Technology": r"(?-i:\b("
    + "|".join(re.escape(t) for t in sorted(TECHNOLOGY_GAZETTEER, key=len, reverse=True))
    + r")\b)",
    "Project": rf"(?-i:\b(Project[ \t]+{_WORD})\b)",
    "Event": rf"(?-i:\b((?:{_WORD}[ \t]+){{1,4}}(?:Summit|Conference|Hackathon))\b)",
    "Team": rf"(?-i:\b((?!The\b)(?:{_WORD}[ \t]+){{1,3}}Team)\b)",
    "Organization": (
        rf"(?-i:\b((?:{_WORD}[ \t]+){{1,2}}"
        r"(?:Artificial[ \t]+Intelligence|AI|Robotics|Labs)"
        r"(?:[ \t]+(?:Inc|Corp|Ltd|LLC)\b\.?)?))"
    ),
    # Product names in this domain are CamelCase single tokens.
    "Product": r"(?-i:\b([A-Z][a-z]+[A-Z][A-Za-z]+)\b)",
    # Semantica reads ``match.group(1)``, so each pattern has exactly one capture.
    "Person": (
        r"(?-i:\b("
        # "Dr. Sarah Chen" / "Sarah" immediately before a person verb ...
        rf"(?:Dr\.[ \t]+)?{_NOT_A_NAME}[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)?"
        rf"(?=\s+{_PERSON_VERBS}\b)"
        # ... or a full name that is the object of "... with".
        rf"|(?<=with ){_NOT_A_NAME}[A-Z][a-z]+[ \t]+[A-Z][a-z]+\b"
        r"))"
    ),
}


def relation_patterns(entity_texts: Iterable[str]) -> dict[str, list[str]]:
    """Build relation and attribute regexes over the entities found in a document."""
    unique = sorted(set(entity_texts), key=len, reverse=True)
    if not unique:
        return {}
    entity = "(?:" + "|".join(re.escape(text) for text in unique) + ")"
    subject = rf"(?<![\w])(?P<subject>{entity})"
    obj = rf"(?:the\s+)?(?P<object>{entity})(?![\w])"

    patterns: dict[str, list[str]] = {}
    for spec in RELATIONS:
        rules = [rf"{subject}\s+(?:{phrase})\s+{obj}" for phrase in spec.phrases]
        # Passive voice: the grammatical subject is the semantic object.
        rules += [
            rf"(?<![\w])(?P<object>{entity})\s+(?:{phrase})\s+(?:the\s+)?(?P<subject>{entity})(?![\w])"
            for phrase in spec.inverse_phrases
        ]
        patterns[spec.name] = rules
    return patterns


def attribute_value_entities_patterns() -> dict[str, str]:
    """Entity patterns for literal attribute values (years)."""
    return {"Year": r"(?<=launched in )((?:19|20)\d{2})\b"}


def attribute_patterns(entity_texts: Iterable[str]) -> dict[str, list[str]]:
    """Regexes for single-valued attributes, expressed as subject/object relations."""
    unique = sorted(set(entity_texts), key=len, reverse=True)
    if not unique:
        return {}
    entity = "(?:" + "|".join(re.escape(text) for text in unique) + ")"
    patterns: dict[str, list[str]] = {}
    for spec in ATTRIBUTES:
        value = spec.value_pattern.replace("(?P<value>", "(?P<object>")
        rules = [
            rf"(?<![\w])(?P<subject>{entity})\s+(?:{phrase})\s+{value}(?![\w])"
            for phrase in spec.phrases
        ]
        rules += [
            rf"{phrase}\s+of\s+(?P<subject>{entity})\s+is\s+(?P<object>{entity})(?![\w])"
            for phrase in spec.prefix_phrases
        ]
        patterns[spec.name] = rules
    return patterns


# "from 2023-01-01 until 2024-12-31" / "since 2025-01-01" after a statement.
_DATE = r"(\d{4}-\d{2}-\d{2})"
VALID_RANGE = re.compile(rf"\bfrom\s+{_DATE}\s+(?:until|to)\s+{_DATE}", re.IGNORECASE)
VALID_SINCE = re.compile(rf"\bsince\s+{_DATE}", re.IGNORECASE)
VALID_UNTIL = re.compile(rf"\buntil\s+{_DATE}", re.IGNORECASE)


def temporal_bounds(sentence: str) -> tuple[str | None, str | None]:
    """Return ``(valid_from, valid_until)`` stated in a sentence, if any."""
    if match := VALID_RANGE.search(sentence):
        return match.group(1), match.group(2)
    since = VALID_SINCE.search(sentence)
    until = VALID_UNTIL.search(sentence)
    return (since.group(1) if since else None, until.group(1) if until else None)
