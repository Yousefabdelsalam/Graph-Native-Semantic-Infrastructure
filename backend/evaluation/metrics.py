"""Metric definitions (application-owned, deterministic, no LLM judge).

* **coverage** – share of expected answer items mentioned anywhere in the evidence
  (evidence text plus source file names).
* **correct** – answerable question: every expected item is present, nothing
  listed as ``excluded`` is present, and the retriever did not declare the
  evidence insufficient.  Unanswerable question: the retriever said so.
* **source recall / precision** – expected supporting documents found, and the
  share of returned documents that were expected.
* **traceability** – share of evidence items that name both a source document and
  a location inside it.

Retrieval metrics say whether the answer is *in* the evidence.  They do not show
that a person or model would extract it correctly; see the LLM column for that.
"""

from __future__ import annotations

import re
from statistics import mean

from backend.schemas import RetrievalResult


def _mentions(text: str, alias: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE) is not None


def coverage(expected: list[list[str]], text: str) -> float | None:
    if not expected:
        return None
    return sum(any(_mentions(text, a) for a in group) for group in expected) / len(expected)


def score_result(question: dict, result: RetrievalResult) -> dict:
    text = "\n".join(f"{e.text} {' '.join(e.filenames)}" for e in result.evidence)
    expected = question.get("expected", [])
    covered = coverage(expected, text)
    leaked = [term for term in question.get("excluded", []) if _mentions(text, term)]
    wanted = set(question.get("sources", []))
    returned = set(result.source_files)
    traceable = [bool(e.filenames and e.location) for e in result.evidence]
    if expected:
        correct = covered == 1.0 and not leaked and result.sufficient
    else:
        correct = not result.sufficient
    return {
        "question_id": question["id"],
        "category": question["category"],
        "mode": result.mode,
        "latency_ms": result.latency_ms,
        "evidence_items": len(result.evidence),
        "evidence_chars": sum(len(e.text) for e in result.evidence),
        "coverage": covered,
        "correct": correct,
        "sufficient": result.sufficient,
        "excluded_leaked": leaked,
        "source_recall": len(wanted & returned) / len(wanted) if wanted else None,
        "source_precision": len(wanted & returned) / len(returned) if wanted and returned else None,
        "traceability": mean(traceable) if traceable else None,
    }


def score_answer(question: dict, answer: str) -> bool:
    """Correctness of a generated answer against the manual expectation."""
    if not question.get("expected"):
        return "insufficient evidence" in answer.lower()
    if any(_mentions(answer, term) for term in question.get("excluded", [])):
        return False
    return coverage(question["expected"], answer) == 1.0


def _average(rows: list[dict], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(mean(values), 3) if values else None


def summarise(rows: list[dict]) -> dict:
    summary = {
        "questions": len(rows),
        "correct": sum(r["correct"] for r in rows),
    }
    for key in (
        "coverage",
        "source_recall",
        "source_precision",
        "traceability",
        "latency_ms",
        "evidence_chars",
        "answer_correct",
    ):
        summary[key] = _average(rows, key)
    return summary


def extraction_scores(
    predicted: set[tuple[str, str, str, str]],
    gold: list[dict],
    symmetric: set[str],
) -> dict:
    """Precision/recall over ``(document, subject, predicate, object)`` tuples."""

    def key(document: str, subject: str, predicate: str, obj: str) -> tuple:
        a, b = subject.lower(), obj.lower()
        if predicate in symmetric:
            a, b = sorted((a, b))
        return document, a, predicate, b

    gold_by_key = {
        key(g["document"], g["subject"], g["predicate"], g["object"]): g for g in gold
    }
    predicted_by_key = {key(*p): p for p in predicted}
    hits = gold_by_key.keys() & predicted_by_key.keys()
    precision = len(hits) / len(predicted_by_key) if predicted_by_key else None
    recall = len(hits) / len(gold_by_key) if gold_by_key else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall
        else 0.0
    )
    return {
        "predicted": len(predicted_by_key),
        "gold": len(gold_by_key),
        "correct": len(hits),
        "precision": None if precision is None else round(precision, 3),
        "recall": None if recall is None else round(recall, 3),
        "f1": round(f1, 3),
        "missed": [gold_by_key[k] for k in sorted(gold_by_key.keys() - hits)],
        "spurious": [predicted_by_key[k] for k in sorted(predicted_by_key.keys() - hits)],
    }
