"""Compare retrieval modes on the sample corpus and refresh the report.

    python scripts/run_evaluation.py [--generate]

Requires the sample data to be loaded (scripts/load_sample_data.py).
``--generate`` also scores Groq-generated answers (needs GROQ_API_KEY).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import configure_logging, load_settings  # noqa: E402
from backend.evaluation.runner import render_results, run_evaluation, write_report  # noqa: E402
from backend.services.lab import KnowledgeLab  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="also score LLM answers")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "evaluation_results.md")
    arguments = parser.parse_args()

    settings = load_settings()
    configure_logging("ERROR")
    if arguments.generate and not settings.llm_available:
        print("--generate needs GROQ_API_KEY", file=sys.stderr)
        return 2
    lab = KnowledgeLab(settings)
    try:
        if not lab.registry.list_documents():
            print("No documents loaded. Run scripts/load_sample_data.py first.", file=sys.stderr)
            return 2
        report = run_evaluation(lab, generate=arguments.generate)
    finally:
        lab.close()
    write_report(report, arguments.output)
    sys.stdout.reconfigure(encoding="utf-8")
    print(render_results(report))
    print(f"Report written to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
