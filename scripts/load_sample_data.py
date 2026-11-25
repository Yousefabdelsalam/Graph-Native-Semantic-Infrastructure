"""Ingest the sample corpus into Neo4j, FAISS, SQLite and the provenance store.

    python scripts/load_sample_data.py [--method regex|pattern|llm] [--reset]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import configure_logging, load_settings  # noqa: E402
from backend.services.decisions import SAMPLE_DECISION, DecisionService  # noqa: E402
from backend.services.lab import KnowledgeLab  # noqa: E402


def record_sample_decision(service: DecisionService) -> None:
    """Add the sample decision unless an intact copy exists.

    A reset removes the entity nodes, which also removes an earlier decision's
    links, so the mere presence of a decision is not enough.
    """
    for decision in service.list_decisions():
        if (
            decision["scenario"] == SAMPLE_DECISION["scenario"]
            and service.trace(decision["decision_id"])["linked_entities"]
        ):
            return
    service.record(**SAMPLE_DECISION)
    print("Recorded the sample decision.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="regex", choices=["regex", "pattern", "llm"])
    parser.add_argument("--reset", action="store_true", help="remove all documents first")
    arguments = parser.parse_args()

    settings = load_settings()
    configure_logging("WARNING")
    lab = KnowledgeLab(settings)
    try:
        lab.repository.connect()
        if arguments.reset:
            lab.reset()
        outcomes = lab.load_sample_data(arguments.method)
        for outcome in outcomes:
            print(
                f"{outcome.status:9} {outcome.filename:30} "
                f"entities={outcome.entities:3} relations={outcome.relationships:3} {outcome.message}"
            )
        if not any(o.status == "failed" for o in outcomes):
            record_sample_decision(DecisionService(lab))
        stats = lab.repository.stats()
        analysis = lab.analysis()
        print(f"\nGraph: {stats['entities']} entities, {stats['relationships']} relationships")
        print(
            f"Conflicts: {len(analysis['conflicts'])}, merges: {len(analysis['merges'])}, "
            f"duplicate candidates: {len(analysis['duplicate_candidates'])}"
        )
        print(f"Vector chunks: {lab.vector.count()}")
        return 1 if any(o.status == "failed" for o in outcomes) else 0
    finally:
        lab.close()


if __name__ == "__main__":
    raise SystemExit(main())
