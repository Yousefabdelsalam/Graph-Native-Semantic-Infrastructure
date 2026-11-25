"""Fingerprint everything the lab persists, to prove it survives a restart.

    python scripts/snapshot.py > before.json
    docker compose down; docker compose up -d --wait neo4j
    python scripts/snapshot.py > after.json     # the two files should be identical
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import configure_logging, load_settings  # noqa: E402
from backend.services.decisions import DecisionService  # noqa: E402
from backend.services.lab import KnowledgeLab  # noqa: E402


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def main() -> None:
    settings = load_settings()
    configure_logging("ERROR")
    lab = KnowledgeLab(settings)
    try:
        graph = lab.graph()
        chunks = lab.vector.chunks()
        provenance = lab.provenance.verify_integrity()
        decisions = DecisionService(lab).list_decisions()
        question = "Which projects use PostgreSQL?"
        result = lab.engine().retrieve(question, "hybrid")
        snapshot = {
            "documents": [
                [d.original_filename, d.file_hash[:16], d.processing_status]
                for d in lab.registry.list_documents()
            ],
            "graph": {
                "entities": len(graph["entities"]),
                "relationships": len(graph["relationships"]),
                "digest": _digest(graph),
            },
            "vectors": {"chunks": len(chunks), "digest": _digest(chunks)},
            "provenance": {
                "entries": provenance["total_entries"],
                "chain_valid": provenance["valid"],
            },
            "decisions": {"count": len(decisions), "digest": _digest(decisions)},
            "conflicts": len(lab.analysis()["conflicts"]),
            "hybrid_query_evidence": _digest([e.reference for e in result.evidence]),
        }
    finally:
        lab.close()
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
