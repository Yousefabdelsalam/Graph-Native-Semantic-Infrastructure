"""Snapshot of Phase 1 behaviour on the original corpus, and the pattern-pack freeze.

``tests/golden/phase1.json`` was recorded from the Phase 1 code before any Phase 2
change.  Rewrite it only for an intended behaviour change, and say so in the commit:

    python -m tests.regression --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from backend.config import Settings
from backend.evaluation.metrics import score_result
from backend.evaluation.runner import (
    evaluate_extraction,
    evaluate_paraphrases,
    load_questions,
)
from backend.extraction import patterns
from backend.ontology.schema import ATTRIBUTES, RELATIONS
from backend.services.lab import KnowledgeLab

GOLDEN = Path(__file__).parent / "golden" / "phase1.json"
MODES = ("vector", "graph", "hybrid", "semantica")
_SCORES = ("predicted", "gold", "correct", "precision", "recall", "f1")


def pattern_fingerprint() -> str:
    """Hash of everything that decides what the ``regex`` method extracts."""
    pack = {
        "entities": patterns.ENTITY_PATTERNS,
        "year": patterns.attribute_value_entities_patterns(),
        "gazetteer": patterns.TECHNOLOGY_GAZETTEER,
        "relations": [
            [r.name, r.domain, r.range, r.phrases, r.inverse_phrases, r.symmetric]
            for r in RELATIONS
        ],
        "attributes": [
            [a.name, a.subject_types, a.phrases, a.prefix_phrases, a.value_pattern]
            for a in ATTRIBUTES
        ],
        # One name only: equal-length names are ordered by set iteration, which
        # varies between processes (without changing what the regex matches).
        "templates": [patterns.relation_patterns(["Alpha"]), patterns.attribute_patterns(["Alpha"])],
        "temporal": [
            patterns.VALID_RANGE.pattern,
            patterns.VALID_SINCE.pattern,
            patterns.VALID_UNTIL.pattern,
        ],
    }
    return hashlib.sha256(json.dumps(pack, sort_keys=True).encode("utf-8")).hexdigest()


def corpus_checksums(settings: Settings) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(settings.sample_dir.iterdir())
        if path.is_file()
    }


def build_snapshot(lab: KnowledgeLab) -> dict:
    """Behaviour of a lab that has exactly the sample corpus loaded."""
    graph = lab.graph()
    analysis = lab.analysis()
    engine = lab.engine()
    retrieval = {}
    for question in load_questions(lab.settings):
        for mode in MODES:
            row = score_result(question, engine.retrieve(question["question"], mode))
            retrieval[f"{question['id']}:{mode}"] = [
                row["correct"],
                row["sufficient"],
                None if row["coverage"] is None else round(row["coverage"], 3),
            ]
    facts = sorted(
        f"{e.filename}|{f.subject}|{f.predicate}|{f.object}"
        for document in lab.registry.list_documents()
        if (e := lab.extraction(document.id))
        for f in e.facts
    )
    scores = {
        name: {method: {k: values[k] for k in _SCORES} for method, values in report.items()}
        for name, report in (
            ("sample", evaluate_extraction(lab.settings, ["pattern", "regex"])),
            ("paraphrase", evaluate_paraphrases(lab.settings, ["pattern", "regex"])),
        )
    }
    return {
        "checksums": corpus_checksums(lab.settings),
        "pattern_fingerprint": pattern_fingerprint(),
        "facts": facts,
        "entities": sorted(e["id"] for e in graph["entities"]),
        "relationships": sorted(
            f"{r['source']}|{r['type']}|{r['target']}" for r in graph["relationships"]
        ),
        "conflicts": sorted(
            f"{c['entity_id']}|{c['property']}|{'/'.join(c['values'])}"
            for c in analysis["conflicts"]
        ),
        "successions": sorted(
            f"{s['entity_id']}|{s['property']}" for s in analysis["successions"]
        ),
        "merges": sorted(f"{m['canonical']}|{'/'.join(m['merged'])}" for m in analysis["merges"]),
        "extraction_scores": scores,
        "retrieval": retrieval,
    }


def main() -> None:
    from tests.conftest import make_settings
    from tests.fakes import InMemoryRepository

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="overwrite the golden file")
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        lab = KnowledgeLab(make_settings(Path(directory)), repository=InMemoryRepository())
        try:
            lab.load_sample_data()
            snapshot = build_snapshot(lab)
        finally:
            lab.close()
    text = json.dumps(snapshot, indent=1, sort_keys=True) + "\n"
    if arguments.write:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {GOLDEN}")
    else:
        print("matches" if GOLDEN.read_text(encoding="utf-8") == text else "DIFFERS")


if __name__ == "__main__":
    main()
