"""Generate the synthetic "Acme AI" sample corpus (TXT, PDF, DOCX, HTML).

Everything here is fictional.  The corpus is designed to exercise:

* overlapping entities across documents (entity resolution, provenance),
* name variants ("Acme AI" / "Acme AI Inc." / "Acme Artificial Intelligence",
  "PostgreSQL" / "Postgres", "Sarah Chen" / "Dr. Sarah Chen" / "Sarah"),
* deliberate contradictions in ``06_legacy_wiki.txt``,
* a time-bounded succession of facts (temporal queries), and
* a few sentences phrased so that pattern extraction is *expected to miss them*
  (pronouns, nominalisations), so extraction recall is measured honestly.

Run with:  python scripts/generate_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_documents"

COMPANY_OVERVIEW = """Acme AI Company Overview

Acme AI is a technology company that builds applied machine learning products.

Employees
Sarah Chen works for Acme AI. David Okafor works for Acme AI.
Priya Raman works for Acme AI. Marcus Webb works for Acme AI.
Elena Rossi works for Acme AI. Tom Nguyen works for Acme AI.

Teams
Dr. Sarah Chen leads the VisionCore Team. Elena Rossi leads the Document Intelligence Team.
The VisionCore Team is part of Acme AI. The Platform Team is part of Acme AI.
The Document Intelligence Team is part of Acme AI.
Tom Nguyen is a member of the VisionCore Team.
Sarah collaborates with Tom Nguyen on model evaluation.

Her deputy joined last spring and now reviews every release.
"""

PRODUCT_PAGES = [
    [
        "Acme AI Product Catalogue",
        "",
        "VisionCore",
        "Acme AI developed VisionCore, a computer vision platform.",
        "VisionCore was launched in 2023.",
        "VisionCore uses PyTorch for model training.",
        "VisionCore uses ONNX Runtime for low latency inference.",
        "The VisionCore Team is responsible for VisionCore.",
        "VisionCore was demonstrated at the Applied Vision Summit.",
    ],
    [
        "DocuMind",
        "Acme AI developed DocuMind, a document intelligence service.",
        "DocuMind was launched in 2022.",
        "DocuMind uses FastAPI for its public interface.",
        "The Document Intelligence Team is responsible for DocuMind.",
        "",
        "FleetSense",
        "Acme AI developed FleetSense, an edge analytics product.",
        "FleetSense uses Kafka to stream telemetry.",
        "The Platform Team is responsible for FleetSense.",
    ],
]

TECH_STACK_SECTIONS = [
    (
        "Datastores",
        [
            "VisionCore depends on PostgreSQL.",
            "The primary database of VisionCore is PostgreSQL.",
            "VisionCore depends on Redis.",
            "DocuMind depends on Postgres.",
            "FleetSense depends on Kafka.",
            "FleetSense depends on Redis.",
        ],
    ),
    (
        "Platform",
        [
            "The Platform Team uses Kubernetes.",
            "Marcus Webb led the Platform Team from 2023-01-01 until 2024-12-31.",
            "Priya Raman leads the Platform Team since 2025-01-01.",
            "Caching for the document service is delegated to an in-memory store "
            "that the platform group operates.",
        ],
    ),
]

PROJECTS_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Acme AI Project Register</title></head>
<body>
<h1>Acme AI Project Register</h1>

<h2>Project Atlas</h2>
<p>Project Atlas is owned by the VisionCore Team.</p>
<p>Project Atlas uses PostgreSQL.</p>
<p>Sarah Chen leads Project Atlas.</p>
<p>Priya Raman collaborates with Sarah Chen on Project Atlas.</p>

<h2>Project Beacon</h2>
<p>Project Beacon is owned by the Document Intelligence Team.</p>
<p>Project Beacon uses Redis.</p>

<h2>Project Cobalt</h2>
<p>Project Cobalt is owned by the Platform Team.</p>
<p>Project Cobalt uses Kubernetes.</p>
<p>Project Cobalt depends on PostgreSQL.</p>
</body>
</html>
"""

INTEGRATIONS = """Acme AI Integrations and Technical Relationships

DocuMind depends on VisionCore for optical character recognition.
FleetSense integrates with VisionCore.
Project Atlas depends on VisionCore.
Acme Artificial Intelligence collaborates with Northwind Robotics Inc. on warehouse automation.
Northwind Robotics Inc. uses FleetSense.
Sarah Chen collaborates with David Okafor on the roadmap.
"""

LEGACY_WIKI = """Archived Engineering Wiki

This page was imported from an old wiki and may be out of date.

Acme AI Inc. developed VisionCore.
VisionCore was launched in 2024.
David Okafor leads the VisionCore Team.
The primary database of VisionCore is MongoDB.
"""


def write_pdf(path: Path, pages: list[list[str]]) -> None:
    """Write a multi-page PDF so that page-level provenance can be tested."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(path), pagesize=A4, invariant=1)
    pdf.setTitle("Acme AI Product Catalogue")
    pdf.setAuthor("Acme AI (synthetic)")
    _, height = A4
    for lines in pages:
        text = pdf.beginText(56, height - 72)
        text.setFont("Helvetica", 11)
        for line in lines:
            text.textLine(line)
        pdf.drawText(text)
        pdf.showPage()
    pdf.save()


def write_docx(path: Path) -> None:
    from docx import Document

    document = Document()
    document.core_properties.title = "Acme AI Technology Stack"
    document.core_properties.author = "Acme AI (synthetic)"
    document.add_heading("Acme AI Technology Stack", level=1)
    for heading, sentences in TECH_STACK_SECTIONS:
        document.add_heading(heading, level=2)
        for sentence in sentences:
            document.add_paragraph(sentence)
    document.save(str(path))


def main(out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "01_company_overview.txt": lambda p: p.write_text(COMPANY_OVERVIEW, encoding="utf-8"),
        "02_product_catalogue.pdf": lambda p: write_pdf(p, PRODUCT_PAGES),
        "03_technology_stack.docx": write_docx,
        "04_project_register.html": lambda p: p.write_text(PROJECTS_HTML, encoding="utf-8"),
        "05_integrations.txt": lambda p: p.write_text(INTEGRATIONS, encoding="utf-8"),
        "06_legacy_wiki.txt": lambda p: p.write_text(LEGACY_WIKI, encoding="utf-8"),
    }
    written: list[Path] = []
    for name, writer in files.items():
        path = out_dir / name
        writer(path)
        written.append(path)
        print(f"wrote {path}")
    return written


if __name__ == "__main__":
    main()
