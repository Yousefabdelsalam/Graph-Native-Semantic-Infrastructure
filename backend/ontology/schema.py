"""The small Acme AI domain vocabulary shared by extraction, graph and ontology code.

This module is pure data (application-owned).  Semantica consumes it:
extraction patterns are built from it, ``OntologyEngine`` renders it to OWL and
SHACL, and ``DuplicateDetector`` results are filtered by its entity types.
"""

from __future__ import annotations

from dataclasses import dataclass

ENTITY_TYPES: tuple[str, ...] = (
    "Person",
    "Organization",
    "Product",
    "Technology",
    "Team",
    "Project",
    "Event",
)


@dataclass(frozen=True)
class RelationSpec:
    """A relationship type with its allowed endpoint types.

    ``phrases`` are regular-expression fragments matched *between* a subject and
    an object.  ``inverse_phrases`` match passive voice, where the text order is
    object-first ("Project Atlas is owned by the VisionCore Team").
    """

    name: str
    description: str
    domain: tuple[str, ...]
    range: tuple[str, ...]
    phrases: tuple[str, ...]
    inverse_phrases: tuple[str, ...] = ()
    symmetric: bool = False
    extension: bool = False  # True when added beyond the seven requested types


RELATIONS: tuple[RelationSpec, ...] = (
    RelationSpec(
        "WORKS_FOR",
        "A person is employed by an organization.",
        ("Person",),
        ("Organization",),
        (r"works?\s+for", r"works?\s+at", r"is\s+employed\s+by"),
    ),
    RelationSpec(
        "LEADS",
        "A person leads a team or project.",
        ("Person",),
        ("Team", "Project"),
        (r"leads", r"led", r"has\s+led", r"heads", r"manages"),
        inverse_phrases=(r"is\s+led\s+by", r"is\s+headed\s+by"),
    ),
    RelationSpec(
        "DEVELOPED",
        "An organization or team built a product.",
        ("Organization", "Team"),
        ("Product",),
        (r"developed", r"built", r"created"),
        inverse_phrases=(r"(?:is|was)\s+(?:developed|built|created)\s+by",),
    ),
    RelationSpec(
        "USES",
        "A product, project, team or organization uses a technology or product.",
        ("Product", "Project", "Team", "Organization"),
        ("Technology", "Product"),
        (r"uses", r"relies\s+on", r"is\s+powered\s+by"),
        inverse_phrases=(r"is\s+used\s+by",),
    ),
    RelationSpec(
        "DEPENDS_ON",
        "A product or project cannot run without another component.",
        ("Product", "Project"),
        ("Technology", "Product"),
        (r"depends\s+on", r"requires"),
    ),
    RelationSpec(
        "PART_OF",
        "Membership: person in team, team in organization.",
        ("Person", "Team", "Project", "Product"),
        ("Team", "Organization", "Project", "Product"),
        (r"is\s+part\s+of", r"is\s+a\s+member\s+of", r"belongs\s+to"),
    ),
    RelationSpec(
        "COLLABORATES_WITH",
        "Two people or organizations work together.",
        ("Person", "Organization", "Team"),
        ("Person", "Organization", "Team"),
        (r"collaborates\s+with", r"works\s+closely\s+with", r"partners\s+with"),
        symmetric=True,
    ),
    RelationSpec(
        "OWNS",
        "A team is accountable for a product or project.",
        ("Team",),
        ("Product", "Project"),
        (r"owns", r"is\s+responsible\s+for"),
        inverse_phrases=(r"is\s+owned\s+by",),
        extension=True,
    ),
    RelationSpec(
        "INTEGRATES_WITH",
        "Two products exchange data or functionality.",
        ("Product",),
        ("Product",),
        (r"integrates\s+with",),
        symmetric=True,
        extension=True,
    ),
)

RELATION_BY_NAME: dict[str, RelationSpec] = {spec.name: spec for spec in RELATIONS}


@dataclass(frozen=True)
class AttributeSpec:
    """A single-valued property stated in text; used for conflict detection."""

    name: str
    description: str
    subject_types: tuple[str, ...]
    phrases: tuple[str, ...]
    # Regex for the value (named group ``value``).
    value_pattern: str
    # Phrase order "The <phrase> of <subject> is <value>".
    prefix_phrases: tuple[str, ...] = ()


ATTRIBUTES: tuple[AttributeSpec, ...] = (
    AttributeSpec(
        "launch_year",
        "Year in which a product was launched.",
        ("Product",),
        (r"was\s+launched\s+in", r"launched\s+in"),
        r"(?P<value>(?:19|20)\d{2})",
    ),
    AttributeSpec(
        "primary_database",
        "The main datastore of a product.",
        ("Product",),
        (),
        r"(?P<value>[A-Z][A-Za-z]+)",
        prefix_phrases=(r"primary\s+database",),
    ),
)

# Relations for which an object can have only one subject at any moment, e.g. a
# team has a single lead.  Overlapping claims are reported as conflicts.
FUNCTIONAL_RELATIONS: dict[str, tuple[str, str]] = {
    # relation -> (property name on the object, object types it applies to)
    "LEADS": ("lead", "Team"),
}

# --- Vocabulary for AI (LLM) extraction -------------------------------------
# Kept separate so the offline regex pattern pack above is unaffected.  General
# documents need ideas as nodes ("planning", "attention") and looser links.

CONCEPT = "Concept"
ALL_ENTITY_TYPES: tuple[str, ...] = (*ENTITY_TYPES, CONCEPT)


def _llm_relation(name: str, description: str, symmetric: bool = False) -> RelationSpec:
    return RelationSpec(
        name,
        description,
        ALL_ENTITY_TYPES,
        ALL_ENTITY_TYPES,
        phrases=(),
        symmetric=symmetric,
        extension=True,
    )


LLM_ONLY_RELATIONS: tuple[RelationSpec, ...] = (
    _llm_relation("IS_A", "The subject is a kind or example of the object."),
    _llm_relation("ENABLES", "The subject provides or makes the object possible."),
    _llm_relation("USED_FOR", "The subject is applied to the object."),
    _llm_relation("EVALUATES", "The subject measures or benchmarks the object."),
    _llm_relation("RELATED_TO", "A stated connection that fits no other type.", symmetric=True),
)
ALL_RELATION_BY_NAME: dict[str, RelationSpec] = {
    **RELATION_BY_NAME,
    **{spec.name: spec for spec in LLM_ONLY_RELATIONS},
}

# Allowed endpoint types for model-extracted relations.  Only relations that
# imply a person or organisation are restricted; the rest accept any pair.
LLM_ENDPOINTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "WORKS_FOR": (("Person",), ("Organization", "Team")),
    "LEADS": (("Person",), ("Team", "Project", "Organization")),
    "DEVELOPED": (
        ("Organization", "Team", "Person"),
        ("Product", "Technology", CONCEPT, "Project"),
    ),
}


def llm_endpoints(relation: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return LLM_ENDPOINTS.get(relation, (ALL_ENTITY_TYPES, ALL_ENTITY_TYPES))
