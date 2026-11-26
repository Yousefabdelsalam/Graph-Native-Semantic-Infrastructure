"""Layout reading, evidence statuses, entity resolution and inferred-link handling.

Everything is synthetic: a small two-column PDF with a table and a running footer
is generated here, and model results are written by hand.  No API is called.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from backend.extraction.validation import (
    find_spans,
    named_together,
    relationship_flags,
    validate,
)
from backend.graph.builder import assemble_graph
from backend.graph.reasoning import infer
from backend.graph.resolution import EntityResolution
from backend.retrieval.answer import build_prompt
from backend.retrieval.engine import INFERRED_LABEL, ONLY_INFERRED_NOTE
from backend.schemas import DocumentExtraction, Fact, Mention
from backend.services.lab import KnowledgeLab
from tests.conftest import make_settings
from tests.fakes import InMemoryRepository

_HEIGHT = A4[1]
FOOTER = "Orbit Handbook"


def _write_pdf(path: Path) -> Path:
    pdf = canvas.Canvas(str(path), pagesize=A4)

    def line(x: float, top: float, text: str, size: float = 10, bold: bool = False) -> None:
        pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        pdf.drawString(x, _HEIGHT - top, text)

    def footer(number: int) -> None:
        line(50, 805, FOOTER, 8)
        line(480, 805, f"Page {number} of 2", 8)

    line(50, 70, "Orbit Systems Field Guide", 20, True)
    line(50, 110, "1. PLANNING METHODS", 13, True)
    line(50, 132, "Orbit Planner relies on explicit task graphs to split large")
    line(50, 146, "goals into smaller steps.")
    line(60, 180, "Greedy Search", 10, True)
    line(320, 180, "Beam Search", 10, True)
    line(60, 198, "Greedy Search keeps only one")
    line(60, 212, "candidate at every step.")
    line(320, 198, "Beam Search keeps several")
    line(320, 212, "candidates in parallel.")
    footer(1)
    pdf.showPage()

    line(50, 70, "2. STORAGE LAYER", 13, True)
    line(50, 92, "Workers call the Ledger API and write each result to disk.")
    # A ruled table.
    xs, ys = (50, 200, 420), (120, 140, 160, 180)
    for x in xs:
        pdf.line(x, _HEIGHT - ys[0], x, _HEIGHT - ys[-1])
    for y in ys:
        pdf.line(xs[0], _HEIGHT - y, xs[-1], _HEIGHT - y)
    for row, cells in enumerate(
        [("Store", "Technology"), ("Event Log", "Kafka topics"), ("Archive", "Parquet files")]
    ):
        for column, cell in enumerate(cells):
            line(xs[column] + 6, ys[row] + 14, cell)
    line(50, 220, "3. LIMITS, QUOTAS AND ALERTS", 13, True)
    line(50, 242, "Each team receives a monthly budget.")
    footer(2)
    pdf.showPage()
    pdf.save()
    return path


@pytest.fixture(scope="module")
def document(processor, tmp_path_factory):
    path = _write_pdf(tmp_path_factory.mktemp("pdf") / "guide.pdf")
    return processor.parse(path, 1, "guide.pdf")


def _fact(number: int, subject: str, predicate: str, obj: str, page: int, **extra) -> Fact:
    values = dict(
        fact_id=f"fact_{number}",
        kind="relation",
        subject=subject,
        subject_type="Concept",
        predicate=predicate,
        object=obj,
        object_type="Concept",
        quote="raw quote",
        location=f"page {page}, chars 0-9",
        start=0,
        end=9,
        confidence=0.9,
        method="llm",
        extracted_at="2026-01-01T00:00:00+00:00",
    )
    return Fact(**{**values, **extra})


def _mention(text: str, page: int, entity_type: str = "Concept", method: str = "llm") -> Mention:
    return Mention(
        text=text,
        entity_type=entity_type,
        start=0,
        end=len(text),
        confidence=0.9,
        method=method,
        location=f"page {page}, chars 0-{len(text)}",
    )


def _extraction(facts: list[Fact], mentions: list[Mention], document_id: int = 1):
    return DocumentExtraction(
        document_id=document_id,
        filename="guide.pdf",
        method="llm",
        extracted_at="2026-01-01T00:00:00+00:00",
        mentions=mentions,
        facts=facts,
    )


# -- layout -----------------------------------------------------------------------
def test_running_footers_are_not_document_text(document) -> None:
    assert FOOTER not in document.text and "Page 1 of 2" not in document.text
    assert {b.text for b in document.blocks if b.kind == "furniture"} >= {FOOTER}


def test_columns_are_read_one_after_the_other(document) -> None:
    assert "Greedy Search keeps only one candidate at every step." in document.text
    assert "Beam Search keeps several candidates in parallel." in document.text
    assert document.text.index("every step.") < document.text.index("Beam Search")


def test_table_rows_headings_pages_and_offsets(document) -> None:
    rows = [b for b in document.blocks if b.kind == "table"]
    assert rows[0].text == "Store: Event Log; Technology: Kafka topics."
    assert [s.label for s in document.segments] == ["page 1", "page 2"]
    assert [b.text for b in document.blocks if b.kind == "title"] == ["Orbit Systems Field Guide"]
    for block in document.blocks:
        if block.start is not None:
            assert document.text[block.start : block.end] == block.text
    column = next(b for b in document.blocks if b.text.startswith("Beam Search keeps"))
    assert [document.blocks[i].text for i in column.path] == [
        "1. PLANNING METHODS",
        "Beam Search",
    ]


# -- statuses ----------------------------------------------------------------------
CASES = [
    # (subject, predicate, object, page, status, rule)
    ("Orbit Planner", "USES", "task graphs", 1, "supported", "same sentence"),
    ("Event Log", "USES", "Kafka topics", 2, "supported", "same table row"),
    ("Workers", "USES", "Ledger API", 2, "supported", "same sentence"),
    ("Greedy Search", "PART_OF", "PLANNING METHODS", 1, "inferred", "subsection of a section"),
    ("Orbit Planner", "USES", "PLANNING METHODS", 1, "inferred", "heading and section text"),
    ("PLANNING METHODS", "PART_OF", "Beam Search", 1, "rejected", "direction contradicts structure"),
    ("Greedy Search", "USES", "Beam Search", 1, "rejected", "not stated together"),
    ("QUOTAS", "RELATED_TO", "ALERTS", 2, "rejected", "heading only"),
    ("Orbit Handbook", "RELATED_TO", "task graphs", 1, "rejected", "header or footer only"),
    ("Orbit Systems", "USES", "task graphs", 1, "rejected", "title only"),
    ("Event Log", "USES", "Parquet files", 2, "rejected", "not stated together"),
]


@pytest.mark.parametrize(("subject", "predicate", "obj", "page", "status", "rule"), CASES)
def test_every_fact_gets_one_status(document, subject, predicate, obj, page, status, rule) -> None:
    report = validate(_extraction([_fact(1, subject, predicate, obj, page)], []), document)
    [verdict] = report.verdicts
    assert (verdict.status, verdict.rule) == (status, rule), verdict.reason
    assert verdict.reason and verdict.original_quote == "raw quote"
    assert len(report.extraction.facts) == (status != "rejected")
    if status != "rejected":
        assert FOOTER not in verdict.quote and verdict.location.startswith(f"page {page}, chars")


def test_validation_does_not_touch_the_raw_extraction(document) -> None:
    raw = _extraction(
        [_fact(1, "Orbit Handbook", "RELATED_TO", "task graphs", 1)],
        [_mention("Orbit Handbook", 1), _mention("task graphs", 1)],
    )
    before = raw.model_dump_json()
    report = validate(raw, document)
    assert raw.model_dump_json() == before
    assert [m.text for m in report.extraction.mentions] == ["task graphs"]
    assert [m["text"] for m in report.excluded_mentions] == ["Orbit Handbook"]


def test_name_matching_rules() -> None:
    assert find_spans("They follow the ReAct loop.", "REACT FRAMEWORK")
    assert not find_spans("The agent can react quickly.", "ReAct")
    assert find_spans("It calls external APIs.", "API")
    assert find_spans("Chain-of-Thought prompting (CoT)", "Chain of Thought (CoT)")
    assert not named_together("Specialised AI agents work here.", "agents", "AI agents")


def test_reversed_duplicates_and_conflicts_are_reported() -> None:
    flags = relationship_flags(
        [
            ("1", "a", "USES", "b"),
            ("2", "b", "PART_OF", "a"),
            ("3", "c", "PART_OF", "d"),
            ("4", "d", "PART_OF", "c"),
            ("5", "e", "RELATED_TO", "f"),
        ]
    )
    assert sorted((f["kind"], tuple(sorted(f["ids"]))) for f in flags) == [
        ("conflict", ("3", "4")),
        ("reversed duplicate", ("1", "2")),
    ]


# -- graph -------------------------------------------------------------------------
def _graph(document, facts: list[Fact]):
    names = {(f.subject_type, f.subject) for f in facts} | {(f.object_type, f.object) for f in facts}
    mentions = [_mention(text, 1, kind) for kind, text in sorted(names)]
    return assemble_graph([validate(_extraction(facts, mentions), document).extraction])


def test_edges_carry_their_status_and_rejected_facts_are_absent(document) -> None:
    assembled = _graph(
        document,
        [
            _fact(1, "Orbit Planner", "USES", "task graphs", 1),
            _fact(2, "Greedy Search", "PART_OF", "PLANNING METHODS", 1),
            _fact(3, "Greedy Search", "USES", "Beam Search", 1),
        ],
    )
    statuses = {r["type"]: r["support_status"] for r in assembled.relationships}
    assert statuses == {"USES": "supported", "PART_OF": "inferred"}
    assert all(r["evidence_support"] == [r["support_status"]] for r in assembled.relationships)
    # Inferred links are not premises for rule reasoning.
    chain = {"entities": assembled.entities, "relationships": assembled.relationships}
    inferred_only = [dict(r, type="DEPENDS_ON") for r in chain["relationships"]]
    assert infer({**chain, "relationships": inferred_only}) == []


def test_vague_link_is_folded_only_into_an_equally_supported_one(document) -> None:
    folded = _graph(
        document,
        [
            _fact(1, "Orbit Planner", "USES", "task graphs", 1),
            _fact(2, "task graphs", "RELATED_TO", "Orbit Planner", 1),
        ],
    )
    [edge] = folded.relationships
    assert edge["type"] == "USES" and len(edge["provenance_ids"]) == 2 and folded.folded

    # A directly stated "related to" must not disappear behind an inferred claim.
    kept = assemble_graph(
        [
            _extraction(
                [
                    _fact(1, "Greedy Search", "PART_OF", "Planner", 1, support="inferred"),
                    _fact(2, "Greedy Search", "RELATED_TO", "Planner", 1, support="supported"),
                ],
                [_mention("Greedy Search", 1), _mention("Planner", 1)],
            )
        ]
    )
    assert len(kept.relationships) == 2 and not kept.folded


# -- entity resolution --------------------------------------------------------------
def _resolve(*mentions: Mention):
    return EntityResolution().resolve([_extraction([], list(mentions))])


def test_spelling_acronym_plural_and_type_variants_merge() -> None:
    resolved = _resolve(
        _mention("Agentic AI", 1, "Technology"),
        _mention("Agentic AI", 2, "Technology"),
        _mention("AGENTIC AI", 2, "Concept"),
        _mention("foundation model", 1, "Technology"),
        _mention("Foundation models", 2, "Technology"),
        _mention("Chain of Thought (CoT)", 1),
        _mention("CoT", 2, "Technology"),
        _mention("multi-agent system", 1),
        _mention("multi agent system", 2),
        _mention("REACT FRAMEWORK", 1, "Technology"),
        _mention("ReAct", 2, "Product", method="regex"),
    )
    entities = {e["name"]: e for e in resolved.entities.values()}
    assert len(entities) == 5
    assert {"Agentic AI", "Chain of Thought (CoT)", "ReAct", "foundation model"} <= set(entities)
    assert entities["Agentic AI"]["observed_types"] == ["Concept", "Technology"]
    assert entities["Agentic AI"]["type"] == "Technology"
    assert entities["ReAct"]["type"] == "Technology"  # the AI-assigned type wins
    assert "REACT FRAMEWORK" in entities["ReAct"]["aliases"]
    assert resolved.resolve("Product", "REACT FRAMEWORK") == entities["ReAct"]["id"]
    assert resolved.resolve("Concept", "foundation models") is not None
    assert {"acronym", "plural form", "across types"} <= {
        word.strip() for merge in resolved.merges for word in merge["method"].split(",")
    }


def test_things_that_only_look_alike_stay_apart() -> None:
    resolved = _resolve(
        _mention("GPT-4", 1, "Product"),
        _mention("GPT-5", 1, "Product"),
        _mention("Atlas", 1, "Team"),
        _mention("Atlas", 1, "Product"),
        _mention("Windows", 1, "Product"),  # no singular attested
        _mention("Llama (70B)", 1, "Product"),  # not an acronym of the name
        _mention("Llama", 1, "Product"),
    )
    assert len(resolved.entities) == 7


# -- retrieval, answers, provenance --------------------------------------------------
@pytest.fixture(scope="module")
def guide_lab(tmp_path_factory):
    """A lab holding the synthetic PDF with model facts supplied by hand."""
    root = tmp_path_factory.mktemp("guide")
    lab = KnowledgeLab(make_settings(root / "data"), repository=InMemoryRepository())
    content = _write_pdf(root / "guide.pdf").read_bytes()
    document = lab.registry.register("guide.pdf", content).document
    facts = [
        _fact(1, "Orbit Planner", "USES", "task graphs", 1),
        _fact(2, "Greedy Search", "PART_OF", "PLANNING METHODS", 1),
        _fact(3, "Orbit Handbook", "RELATED_TO", "task graphs", 1),
    ]
    mentions = [
        _mention(name, 1)
        for name in (
            "Orbit Planner",
            "task graphs",
            "Greedy Search",
            "PLANNING METHODS",
            "Orbit Handbook",
        )
    ]
    extraction = _extraction(facts, mentions, document.id)
    lab.settings.extractions_dir.mkdir(parents=True, exist_ok=True)
    path = lab.settings.extractions_dir / f"{document.id}.json"
    path.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")
    lab.provenance.record_document(lab.layout(document.id), "hash")
    lab.provenance.record_extraction(extraction)
    raw = path.read_bytes()
    lab.rebuild_graph()
    lab.refresh_search_index(document.id)
    assert path.read_bytes() == raw  # a rebuild never rewrites the raw result
    yield lab
    lab.close()


def test_rebuild_saves_derived_files_separately(guide_lab) -> None:
    assert (guide_lab.settings.layout_dir / "1.json").exists()
    assert (guide_lab.settings.validation_dir / "1.json").exists()
    assert all(FOOTER not in c["text"] for c in guide_lab.vector.chunks())
    assert "Orbit Handbook" not in {e["name"] for e in guide_lab.graph()["entities"]}


def test_provenance_keeps_status_and_history(guide_lab) -> None:
    supported = guide_lab.provenance.get("fact_1")
    inferred = guide_lab.provenance.get("fact_2")
    rejected = guide_lab.provenance.get("fact_3")
    assert supported["metadata"]["support_status"] == "supported"
    assert inferred["metadata"]["support_status"] == "inferred"
    assert inferred["source_quote"].startswith("[Heading] 1. PLANNING METHODS")
    assert rejected["invalidated"] is True
    assert rejected["metadata"]["support_status"] == "rejected"
    assert "raw quote" in str(guide_lab.provenance.lineage("fact_1"))  # earlier version

    before = guide_lab.provenance.statistics()
    guide_lab.rebuild_graph()  # unchanged verdicts write nothing new
    assert guide_lab.provenance.statistics() == before
    assert guide_lab.provenance.verify_integrity()["valid"]


@pytest.mark.parametrize("mode", ["graph", "hybrid", "semantica"])
def test_retrieval_labels_inferred_facts(guide_lab, mode) -> None:
    result = guide_lab.engine().retrieve("What is Greedy Search part of?", mode)
    labelled = [e for e in result.evidence if INFERRED_LABEL in e.text]
    assert labelled
    for item in result.evidence:
        if item.kind == "fact" and "Greedy Search PART_OF" in item.text:
            assert item.support == "inferred" and item.text.startswith(INFERRED_LABEL)
        if item.kind == "fact" and "Orbit Planner USES" in item.text:
            assert item.support == "supported" and INFERRED_LABEL not in item.text


def test_inferred_facts_alone_are_not_sufficient(guide_lab) -> None:
    engine = guide_lab.engine()
    only = {
        key: value
        for key, value in engine.relationships.items()
        if value["support_status"] == "inferred"
    }
    engine.relationships = only
    result = engine.retrieve("What is Greedy Search part of?", "graph")
    assert result.evidence and not result.sufficient
    assert result.note.startswith(ONLY_INFERRED_NOTE)


def test_answer_prompt_forbids_stating_inferences_as_facts(guide_lab) -> None:
    result = guide_lab.engine().retrieve("What is Greedy Search part of?", "graph")
    prompt = build_prompt(result)
    assert "Never state it as a fact" in prompt and INFERRED_LABEL in prompt


def test_rule_based_sample_facts_are_all_directly_supported(loaded_lab) -> None:
    reports = sorted(loaded_lab.settings.validation_dir.glob("*.json"))
    verdicts = [v for p in reports for v in json.loads(p.read_text(encoding="utf-8"))["verdicts"]]
    assert verdicts and {v["status"] for v in verdicts} == {"supported"}
    assert {r["support_status"] for r in loaded_lab.graph()["relationships"]} == {"supported"}
