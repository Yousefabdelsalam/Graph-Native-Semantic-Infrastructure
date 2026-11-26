"""Run the comparison on the sample corpus and render the report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import semantica

from backend.config import Settings
from backend.evaluation.metrics import extraction_scores, score_answer, score_result, summarise
from backend.extraction.extractor import EXTRACTION_METHODS, ExtractionPipeline
from backend.graph.resolution import EntityResolution
from backend.ingestion.parser import DocumentProcessor
from backend.ontology.schema import RELATIONS
from backend.retrieval.answer import AnswerUnavailableError, generate_answer
from backend.retrieval.engine import MODES, RetrievalEngine
from backend.schemas import ParsedDocument, Segment
from backend.services.lab import KnowledgeLab

TOP_K_SWEEP = (1, 2, 3, 4)
MAX_FACTS_SWEEP = (4, 8, 12, 20)


def load_questions(settings: Settings) -> list[dict]:
    path = settings.sample_dir / "evaluation_questions.json"
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def corpus_mismatch(lab: KnowledgeLab) -> str:
    """Expected answers only make sense for exactly the sample documents."""
    expected = {p.name for p in lab.settings.sample_dir.iterdir() if p.suffix != ".json"}
    actual = {
        d.original_filename
        for d in lab.registry.list_documents()
        if d.processing_status == "processed"
    }
    if actual == expected:
        return ""
    return (
        f"The knowledge base differs from the sample corpus (missing: {sorted(expected - actual)}, "
        f"extra: {sorted(actual - expected)}); scores are not comparable."
    )


def evaluate_extraction(settings: Settings, methods: list[str] | None = None) -> dict:
    """Extraction quality per method, independent of what is stored."""
    gold = json.loads((settings.sample_dir / "gold_facts.json").read_text(encoding="utf-8"))
    processor = DocumentProcessor()
    files = sorted(p for p in settings.sample_dir.iterdir() if p.suffix != ".json")
    documents = [processor.parse(path, i, path.name) for i, path in enumerate(files, start=1)]
    methods = methods or ["pattern", "regex"] + (["llm"] if settings.llm_available else [])
    symmetric = {spec.name for spec in RELATIONS if spec.symmetric}
    report = {}
    for method in methods:
        pipeline = ExtractionPipeline(settings, method)
        extractions = [pipeline.extract(document) for document in documents]
        resolved = EntityResolution().resolve(extractions)
        predicted = set()
        for extraction in extractions:
            for fact in extraction.facts:
                if fact.kind != "relation":
                    continue
                subject = resolved.resolve(fact.subject_type, fact.subject)
                obj = resolved.resolve(fact.object_type, fact.object)
                if subject and obj:
                    predicted.add(
                        (
                            extraction.filename,
                            resolved.entities[subject]["name"],
                            fact.predicate,
                            resolved.entities[obj]["name"],
                        )
                    )
        scores = extraction_scores(predicted, gold["facts"], symmetric)
        discarded: dict[str, int] = {}
        for extraction in extractions:
            for reason, count in extraction.discarded.items():
                discarded[reason] = discarded.get(reason, 0) + count
        scores["discarded"] = discarded
        scores["description"] = EXTRACTION_METHODS[method]
        report[method] = scores
    return report


def evaluate_paraphrases(settings: Settings, methods: list[str] | None = None) -> dict:
    """Recall and precision on rewordings the pattern pack was not tuned to."""
    path = settings.sample_dir / "paraphrase_probe.json"
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    methods = methods or ["pattern", "regex"] + (["llm"] if settings.llm_available else [])
    symmetric = {spec.name for spec in RELATIONS if spec.symmetric}
    resolver = EntityResolution()
    report = {}
    for method in methods:
        pipeline = ExtractionPipeline(settings, method)
        predicted: set[tuple[str, str, str, str]] = set()
        gold: list[dict] = []
        for number, case in enumerate(cases, start=1):
            name = f"case {number:02d}"
            document = ParsedDocument(
                document_id=number,
                filename=name,
                file_type="txt",
                text=case["sentence"],
                segments=[Segment(label="body", start=0, end=len(case["sentence"]))],
                parser="paraphrase probe",
            )
            for fact in pipeline.extract(document).facts:
                if fact.kind == "relation":
                    predicted.add(
                        (
                            name,
                            resolver.normalized_name(fact.subject, fact.subject_type),
                            fact.predicate,
                            resolver.normalized_name(fact.object, fact.object_type),
                        )
                    )
            gold += [
                {
                    "document": name,
                    "subject": s,
                    "predicate": p,
                    "object": o,
                    "sentence": case["sentence"],
                }
                for s, p, o in case["facts"]
            ]
        report[method] = extraction_scores(predicted, gold, symmetric)
    return report


def evaluate_retrieval(
    engine: RetrievalEngine,
    settings: Settings,
    questions: list[dict],
    *,
    top_k: int = 4,
    max_facts: int = 12,
    generate: bool = False,
) -> dict:
    # One unmeasured call per mode so model loading does not distort latency.
    for mode in MODES:
        engine.retrieve(questions[0]["question"], mode, top_k=top_k, max_facts=max_facts)
    rows = []
    for question in questions:
        for mode in MODES:
            result = engine.retrieve(question["question"], mode, top_k=top_k, max_facts=max_facts)
            row = score_result(question, result)
            row["answer"] = None
            row["answer_correct"] = None
            if generate:
                try:
                    row["answer"] = generate_answer(settings, result)
                    row["answer_correct"] = score_answer(question, row["answer"])
                except AnswerUnavailableError as exc:
                    row["answer"] = f"(not generated: {exc})"
            rows.append(row)

    by_mode = {m: summarise([r for r in rows if r["mode"] == m]) for m in MODES}
    categories = sorted({q["category"] for q in questions})
    by_category = {
        category: {
            m: summarise([r for r in rows if r["mode"] == m and r["category"] == category])
            for m in MODES
        }
        for category in categories
    }
    return {
        "rows": rows,
        "by_mode": by_mode,
        "by_category": by_category,
        "sensitivity": _sensitivity(engine, questions),
        "settings": {"top_k": top_k, "max_facts": max_facts, "generate": generate},
    }


def _sensitivity(engine: RetrievalEngine, questions: list[dict]) -> list[dict]:
    """How coverage moves with the evidence budget (answerable questions only)."""
    answerable = [q for q in questions if q["expected"]]
    sweeps = [("vector", "top_k", k) for k in TOP_K_SWEEP]
    sweeps += [("graph", "max_facts", n) for n in MAX_FACTS_SWEEP]
    table = []
    for mode, parameter, value in sweeps:
        rows = [
            score_result(q, engine.retrieve(q["question"], mode, **{parameter: value}))
            for q in answerable
        ]
        table.append({"mode": mode, "parameter": parameter, "value": value, **summarise(rows)})
    return table


def run_evaluation(lab: KnowledgeLab, *, generate: bool = False) -> dict:
    questions = load_questions(lab.settings)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "semantica_version": semantica.__version__,
        "embedding_model": lab.settings.embedding_model,
        "llm_model": lab.settings.openai_model if generate else None,
        "warning": corpus_mismatch(lab),
        "corpus": {
            "chunks": lab.vector.count(),
            "entities": len(lab.graph()["entities"]),
            "relationships": len(lab.graph()["relationships"]),
        },
        "questions": questions,
        "retrieval": evaluate_retrieval(
            lab.engine(), lab.settings, questions, generate=generate
        ),
        "extraction": evaluate_extraction(lab.settings),
        "paraphrases": evaluate_paraphrases(lab.settings),
    }
    path = lab.settings.exports_dir / "evaluation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# --------------------------------------------------------------------- markdown
def _cell(value: object) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}" if abs(value) < 10 else f"{value:.0f}"
    return str(value)


def _table(header: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def render_results(report: dict) -> str:
    """The measured part of ``docs/evaluation_results.md`` (no interpretation)."""
    retrieval = report["retrieval"]
    modes = list(MODES)
    out = [
        f"Run: {report['generated_at']} · Semantica {report['semantica_version']} · "
        f"embeddings `{report['embedding_model']}` · top_k={retrieval['settings']['top_k']}, "
        f"max_facts={retrieval['settings']['max_facts']} · corpus: "
        f"{report['corpus']['chunks']} chunks, {report['corpus']['entities']} entities, "
        f"{report['corpus']['relationships']} relationships.",
        f"LLM answers: {'model ' + report['llm_model'] if report['llm_model'] else 'not run'}.",
    ]
    if report["warning"]:
        out.append(f"**Warning:** {report['warning']}")
    columns = [
        ("Correct", "correct"),
        ("Coverage", "coverage"),
        ("Source recall", "source_recall"),
        ("Source precision", "source_precision"),
        ("Traceability", "traceability"),
        ("Latency ms (mean)", "latency_ms"),
        ("Evidence chars (mean)", "evidence_chars"),
        ("LLM answer correct", "answer_correct"),
    ]
    out += ["", "### Overall", ""]
    rows = []
    for mode in modes:
        s = retrieval["by_mode"][mode]
        rows.append(
            [mode]
            + [f"{s['correct']}/{s['questions']}" if k == "correct" else s[k] for _, k in columns]
        )
    out.append(_table(["Mode", *[c for c, _ in columns]], rows))

    out += ["", "### Correct answers by question category", ""]
    rows = [
        [category]
        + [f"{summary[m]['correct']}/{summary[m]['questions']}" for m in modes]
        for category, summary in retrieval["by_category"].items()
    ]
    out.append(_table(["Category", *modes], rows))

    out += ["", "### Per question (✓ correct, ✗ not; coverage in brackets)", ""]
    indexed = {(r["question_id"], r["mode"]): r for r in retrieval["rows"]}
    rows = []
    for question in report["questions"]:
        row = [question["id"], question["category"], question["question"]]
        for mode in modes:
            result = indexed[(question["id"], mode)]
            mark = "✓" if result["correct"] else "✗"
            detail = "" if result["coverage"] is None else f" ({result['coverage']:.2f})"
            if result["excluded_leaked"]:
                detail += " leak"
            if question["expected"] and not result["sufficient"]:
                detail += " refused"
            row.append(mark + detail)
        rows.append(row)
    out.append(_table(["Id", "Category", "Question", *modes], rows))

    out += ["", "### Sensitivity to the evidence budget (answerable questions)", ""]
    out.append(
        _table(
            ["Mode", "Parameter", "Value", "Coverage", "Source precision", "Evidence chars"],
            [
                [
                    s["mode"],
                    s["parameter"],
                    s["value"],
                    s["coverage"],
                    s["source_precision"],
                    s["evidence_chars"],
                ]
                for s in retrieval["sensitivity"]
            ],
        )
    )

    out += ["", "### Extraction quality against the labelled facts", ""]
    out.append(
        _table(
            ["Method", "Predicted", "Gold", "Correct", "Precision", "Recall", "F1"],
            [
                [m, s["predicted"], s["gold"], s["correct"], s["precision"], s["recall"], s["f1"]]
                for m, s in report["extraction"].items()
            ],
        )
    )
    out += ["", "### Paraphrase probe (wording the patterns were not tuned to)", ""]
    out.append(
        _table(
            ["Method", "Predicted", "Gold", "Correct", "Precision", "Recall", "F1"],
            [
                [m, s["predicted"], s["gold"], s["correct"], s["precision"], s["recall"], s["f1"]]
                for m, s in report["paraphrases"].items()
            ],
        )
    )
    for method, scores in report["paraphrases"].items():
        if scores["correct"] and scores["missed"]:
            out += ["", f"Probe sentences `{method}` failed on:", ""]
            out += [f"- {s}" for s in dict.fromkeys(g["sentence"] for g in scores["missed"])]
    for method, scores in report["extraction"].items():
        if scores["predicted"] and scores["missed"]:
            out += ["", f"Facts missed by `{method}`:", ""]
            out += [
                f"- {g['document']}: {g['subject']} {g['predicate']} {g['object']}"
                + (" *(labelled hard)*" if g.get("hard") else "")
                for g in scores["missed"]
            ]
        if scores["spurious"]:
            out += ["", f"Facts `{method}` produced that are not in the labels:", ""]
            out += [f"- {p[0]}: {p[1]} {p[2]} {p[3]}" for p in scores["spurious"]]
    return "\n".join(out) + "\n"


BEGIN = "<!-- BEGIN MEASURED RESULTS -->"
END = "<!-- END MEASURED RESULTS -->"


def write_report(report: dict, path: Path) -> None:
    """Replace the measured block, preserving the hand-written analysis around it."""
    block = f"{BEGIN}\n\n{render_results(report)}\n{END}"
    if path.exists() and BEGIN in (text := path.read_text(encoding="utf-8")):
        head, rest = text.split(BEGIN, 1)
        tail = rest.split(END, 1)[1]
        text = head + block + tail
    else:
        text = f"# Evaluation results\n\n{block}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
