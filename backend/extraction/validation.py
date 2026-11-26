"""Evidence validation: is each extracted relationship stated by the document?

Deterministic rules over the document's blocks; no model is called.  Every fact
receives exactly one status:

``supported``
    Both names appear in one sentence of body text, or in one table row.
``inferred``
    The link comes from the document structure: one name is in a section heading
    and the other in the text of that section, or a subsection is linked to its
    parent section (``PART_OF`` / ``IS_A`` / ``RELATED_TO`` only).  Inferred
    relationships are never to be presented as confirmed facts.
``rejected``
    Anything else: names that occur only in running headers or footers, only in
    a title or heading, or that the relevant passage never brings together.

``supported`` means the two names are stated together.  Whether the sentence
really expresses the predicate (and is not negated) is not something these rules
can decide; it still needs a human reader.

The extraction passed in is never modified.  :func:`validate` returns a copy with
rejected facts and header/footer-only mentions removed, plus one verdict per fact.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from backend.extraction.sentences import split_sentences, unwrap_lines
from backend.graph.vocabulary import ALIASES
from backend.schemas import Block, DocumentExtraction, Fact, ParsedDocument

RULES_VERSION = "1"
SUPPORTED, INFERRED, REJECTED = "supported", "inferred", "rejected"
# Predicates that a heading hierarchy can express.
STRUCTURAL_PREDICATES = frozenset({"PART_OF", "IS_A", "RELATED_TO"})
_HEADINGS = ("heading", "title")
_ACRONYM_SUFFIX = re.compile(r"^(.*\S)\s*\(([A-Za-z][A-Za-z0-9\-]{1,7})\)$")


@dataclass
class Verdict:
    fact_id: str
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    status: str
    rule: str
    reason: str
    quote: str  # validated evidence; the original quote when rejected
    location: str
    start: int
    end: int
    original_quote: str
    original_location: str
    confidence: float


@dataclass
class ValidationReport:
    extraction: DocumentExtraction  # working copy: accepted facts only
    verdicts: list[Verdict]
    excluded_mentions: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rules_version": RULES_VERSION,
            "document_id": self.extraction.document_id,
            "filename": self.extraction.filename,
            "method": self.extraction.method,
            "verdicts": [asdict(v) for v in self.verdicts],
            "excluded_mentions": self.excluded_mentions,
        }


# ----------------------------------------------------------------------- names
def name_forms(name: str) -> list[tuple[str, bool]]:
    """Spellings that count as naming ``name``: ``(form, case_sensitive)``.

    Covers "Long Name (ACR)", singular / plural of the last word and the curated
    alias table.  Short mixed-case names and acronyms ("ReAct", "CoT", "API")
    match case-sensitively, so that "ReAct" is not found in the verb "react".
    """
    base = " ".join(name.split())
    forms = {base}
    if match := _ACRONYM_SUFFIX.match(base):
        forms |= {match.group(1), match.group(2)}
    canonical = ALIASES.get(base.lower()) or next(
        (c for c in ALIASES.values() if c.lower() == base.lower()), None
    )
    if canonical:
        forms |= {canonical, *(a for a, c in ALIASES.items() if c == canonical)}
    for form in list(forms):
        forms |= _number_variants(form)
    return sorted(((f, _case_sensitive(f)) for f in forms), key=lambda f: (-len(f[0]), f[0]))


def _case_sensitive(form: str) -> bool:
    stem = form[:-1] if form.endswith("s") else form
    return " " not in stem and len(stem) <= 6 and sum(c.isupper() for c in stem) >= 2


def _number_variants(form: str) -> set[str]:
    word = form.split(" ")[-1]
    head = form[: len(form) - len(word)]
    lower = word.lower()
    if len(word) < 3:
        return set()
    if len(word) > 3 and word[-1] == "s" and word[-2].isupper():  # "APIs"
        return {head + word[:-1]}
    if word.isupper() and len(word) <= 6:  # "API" -> "APIs"
        return {head + word + "s"}
    if lower.endswith("ies") and len(word) > 4:
        return {head + word[:-3] + "y"}
    if lower.endswith(("sses", "xes", "ches", "shes")):
        return {head + word[:-2]}
    if lower.endswith("s") and not lower.endswith(("ss", "us", "is")):
        return {head + word[:-1]}
    if not lower.endswith("s"):
        return {head + word + "s"}
    return set()


def find_spans(text: str, name: str) -> list[tuple[int, int]]:
    """Whole-word occurrences of any form of ``name`` (hyphens equal spaces)."""
    spans: set[tuple[int, int]] = set()
    for form, sensitive in name_forms(name):
        words = [re.escape(w) for w in re.split(r"[\s\-]+", form) if w]
        if not words:
            continue
        pattern = r"(?<!\w)" + r"[\s\-]+".join(words) + r"(?!\w)"
        flags = 0 if sensitive else re.IGNORECASE
        spans |= {m.span() for m in re.finditer(pattern, text, flags)}
    return sorted(spans)


def named_together(text: str, first: str, second: str) -> bool:
    """Both names occur, in places that do not overlap ("agent" in "AI agent")."""
    others = find_spans(text, second)
    return any(
        a[1] <= b[0] or b[1] <= a[0] for a in find_spans(text, first) for b in others
    )


def _names(text: str, name: str) -> bool:
    return bool(find_spans(text, name))


# ---------------------------------------------------------------------- blocks
def blocks_of(document: ParsedDocument) -> list[Block]:
    """The layout blocks; for non-PDF files, one block per (re-joined) line of text.

    DOCX section headings are known from the segments; plain text and HTML carry
    no heading information, so nothing can be inferred from structure there.
    """
    if document.blocks:
        return document.blocks
    blocks: list[Block] = []
    for segment in document.segments:
        heading: int | None = None
        body = unwrap_lines(document.text[segment.start : segment.end])
        for index, line in enumerate(re.finditer(r"[^\n]+", body)):
            is_heading = index == 0 and segment.label.startswith("section ")
            blocks.append(
                Block(
                    segment=segment.label,
                    kind="heading" if is_heading else "text",
                    text=line.group(0),
                    start=segment.start + line.start(),
                    end=segment.start + line.end(),
                    level=1 if is_heading else 0,
                    path=[] if is_heading or heading is None else [heading],
                )
            )
            if is_heading:
                heading = len(blocks) - 1
    return blocks


def _segment_label(location: str) -> str:
    return location.rsplit(", chars ", 1)[0]


def _units(block: Block) -> list[tuple[int, int, str]]:
    """Statements of a content block: sentences, or the whole table row."""
    if block.kind == "table":
        return [(block.start, block.end, block.text)]
    return [(block.start + s, block.start + e, text) for s, e, text in split_sentences(block.text)]


# ------------------------------------------------------------------ validation
def validate(extraction: DocumentExtraction, document: ParsedDocument) -> ValidationReport:
    blocks = blocks_of(document)
    verdicts = [_judge(fact, blocks) for fact in extraction.facts]
    by_id = {v.fact_id: v for v in verdicts}
    accepted: list[Fact] = []
    for fact in extraction.facts:
        verdict = by_id[fact.fact_id]
        if verdict.status == REJECTED:
            continue
        accepted.append(
            fact.model_copy(
                update={
                    "quote": verdict.quote,
                    "location": verdict.location,
                    "start": verdict.start,
                    "end": verdict.end,
                    "support": verdict.status,
                    "support_reason": verdict.reason,
                }
            )
        )

    kept, furniture = [], []
    for mention in extraction.mentions:
        here = [b for b in blocks if b.segment == _segment_label(mention.location)]
        kinds = {b.kind for b in here if _names(b.text, mention.text)}
        (furniture if kinds == {"furniture"} else kept).append(mention)

    # An accepted fact must keep at least one mention for each of its ends.
    needed = {(f.subject_type, f.subject.lower()) for f in accepted}
    needed |= {(f.object_type, f.object.lower()) for f in accepted}
    needed -= {(m.entity_type, m.text.lower()) for m in kept}
    excluded = []
    for mention in furniture:
        key = (mention.entity_type, mention.text.lower())
        if key in needed:
            needed.remove(key)
            kept.append(mention)
            continue
        label = _segment_label(mention.location)
        excluded.append(
            {
                "text": mention.text,
                "entity_type": mention.entity_type,
                "location": mention.location,
                "reason": f"on {label} it appears only in the running header or footer",
            }
        )
    kept.sort(key=lambda m: m.start)
    copy = extraction.model_copy(update={"mentions": kept, "facts": accepted})
    return ValidationReport(copy, verdicts, excluded)


def _judge(fact: Fact, blocks: list[Block]) -> Verdict:
    label = _segment_label(fact.location)
    content = [b for b in blocks if b.segment == label and b.kind in ("text", "table")]
    subject, obj = fact.subject, fact.object

    def verdict(status: str, rule: str, reason: str, evidence: tuple | None = None) -> Verdict:
        quote, start, end, location = evidence or (
            fact.quote,
            fact.start,
            fact.end,
            fact.location,
        )
        return Verdict(
            fact_id=fact.fact_id,
            subject=subject,
            subject_type=fact.subject_type,
            predicate=fact.predicate,
            object=obj,
            object_type=fact.object_type,
            status=status,
            rule=rule,
            reason=reason,
            quote=quote,
            location=location,
            start=start,
            end=end,
            original_quote=fact.quote,
            original_location=fact.location,
            confidence=fact.confidence,
        )

    # 1. Stated: one sentence or table row names both.
    stated = [
        (block, unit)
        for block in content
        for unit in _units(block)
        if named_together(unit[2], subject, obj)
    ]
    if stated:
        original = " ".join(fact.quote.split())
        block, (start, end, text) = next(
            (item for item in stated if item[1][2] in original or original in item[1][2]),
            stated[0],
        )
        place = "table row" if block.kind == "table" else "sentence"
        return verdict(
            SUPPORTED,
            f"same {place}",
            f"'{subject}' and '{obj}' are named in the same {place}.",
            (text, start, end, f"{label}, chars {start}-{end}"),
        )

    # In a hierarchy "A PART_OF / IS_A B" runs from the narrower to the broader term,
    # so the subject can never be the heading above the object.
    directed = fact.predicate in STRUCTURAL_PREDICATES - {"RELATED_TO"}
    reversed_direction: str | None = None

    # 2a. Inferred: a subsection linked to the section that contains it.
    for block in content:
        if len(block.path) < 2 or fact.predicate not in STRUCTURAL_PREDICATES:
            continue
        parent, child = blocks[block.path[-2]], blocks[block.path[-1]]
        forward = _names(child.text, subject) and _names(parent.text, obj)
        backward = _names(child.text, obj) and _names(parent.text, subject)
        if not (forward or backward):
            continue
        if directed and not forward:
            reversed_direction = (
                f"'{obj}' is a subsection of '{subject}', so '{subject}' {fact.predicate} "
                f"'{obj}' runs against the document structure."
            )
            continue
        start, end, text = _units(block)[0]
        return verdict(
            INFERRED,
            "subsection of a section",
            f"'{child.text}' is a subsection of '{parent.text}'. The link follows from the "
            "document structure, not from a statement.",
            (
                f"[Heading] {parent.text} [Subheading] {child.text} [Text] {text}",
                start,
                end,
                f"{label}, chars {start}-{end}; heading chars {parent.start}-{parent.end}",
            ),
        )

    # 2b. Inferred: one name in an enclosing heading, the other in that section.
    for block in content:
        for index in reversed(block.path):
            heading = blocks[index]
            for in_heading, in_text in ((subject, obj), (obj, subject)):
                if not _names(heading.text, in_heading) or _names(heading.text, in_text):
                    continue
                for start, end, text in _units(block):
                    if not _names(text, in_text):
                        continue
                    if directed and in_heading == subject:
                        reversed_direction = reversed_direction or (
                            f"'{subject}' is the heading above the text naming '{obj}', so "
                            f"'{subject}' {fact.predicate} '{obj}' runs against the document "
                            "structure."
                        )
                        continue
                    return verdict(
                        INFERRED,
                        "heading and section text",
                        f"'{in_heading}' is named by the heading '{heading.text}'; "
                        f"'{in_text}' appears in the text under it. The document does "
                        "not state the link in one sentence.",
                        (
                            f"[Heading] {heading.text} [Text] {text}",
                            start,
                            end,
                            f"{label}, chars {start}-{end}; heading chars "
                            f"{heading.start}-{heading.end}",
                        ),
                    )

    # 3. Rejected, with the most specific reason.
    here = [b for b in blocks if b.segment == label]
    for name in (subject, obj):
        found = {b.kind for b in here if _names(b.text, name)}
        if found == {"furniture"}:
            return verdict(
                REJECTED,
                "header or footer only",
                f"On {label} '{name}' appears only in the running header or footer.",
            )
    if reversed_direction:
        return verdict(REJECTED, "direction contradicts structure", reversed_direction)
    shared = next(
        (b for b in here if b.kind in _HEADINGS and named_together(b.text, subject, obj)), None
    )
    if shared is not None:
        return verdict(
            REJECTED,
            f"{shared.kind} only",
            f"Both names appear only together in the {shared.kind} '{shared.text}'; no body "
            "text states the link.",
        )
    for name in (subject, obj):
        found = {b.kind for b in here if _names(b.text, name)}
        if not found:
            return verdict(REJECTED, "name not found", f"'{name}' is not named on {label}.")
        if found <= {"title", "furniture"}:
            return verdict(
                REJECTED,
                "title only",
                f"On {label} '{name}' is named only in the document title.",
            )
    return verdict(
        REJECTED,
        "not stated together",
        f"'{subject}' and '{obj}' are not named in the same sentence or table row, and no "
        "section heading connects them.",
    )


# ------------------------------------------------------- relationship checks
# Pairs of predicates that state one dependency from both ends.
_INVERSES = (("USES", "PART_OF"), ("DEPENDS_ON", "PART_OF"))
_ASYMMETRIC = frozenset(
    {"USES", "PART_OF", "IS_A", "ENABLES", "DEPENDS_ON", "USED_FOR", "EVALUATES", "LEADS", "OWNS"}
)


def relationship_flags(relationships: list[tuple[str, str, str, str]]) -> list[dict]:
    """Reversed duplicates and two-way contradictions.

    ``relationships`` holds ``(id, source, predicate, target)``.  Nothing is
    removed: both sides are reported for a human to decide.
    """
    index: dict[tuple[str, str, str], list[str]] = {}
    for identifier, source, predicate, target in relationships:
        index.setdefault((source, predicate, target), []).append(identifier)
    flags: list[dict] = []
    for (source, predicate, target), identifiers in sorted(index.items()):
        reverse = index.get((target, predicate, source))
        if predicate in _ASYMMETRIC and reverse and source < target:
            flags.append(
                {
                    "kind": "conflict",
                    "ids": [*identifiers, *reverse],
                    "detail": f"{source} {predicate} {target} and the reverse are both claimed.",
                }
            )
        for first, second in _INVERSES:
            if predicate == first and (opposite := index.get((target, second, source))):
                flags.append(
                    {
                        "kind": "reversed duplicate",
                        "ids": [*identifiers, *opposite],
                        "detail": f"{source} {first} {target} restates {target} {second} {source}.",
                    }
                )
    return flags
