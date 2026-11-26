"""Page 2: upload, inspect, reprocess and delete documents."""

import json

import streamlit as st

from app.components.common import get_lab, require_graph, table
from backend.config import SUPPORTED_EXTENSIONS
from backend.extraction.extractor import EXTRACTION_METHODS
from backend.ingestion.validation import UploadValidationError

lab = get_lab()
st.title("Document Management")
require_graph(lab)  # processing needs Neo4j; fail early with instructions

methods = [m for m in EXTRACTION_METHODS if m != "llm" or lab.settings.llm_available]
method = st.selectbox(
    "Extraction method",
    methods,
    index=methods.index("llm" if "llm" in methods else "regex"),
    format_func=lambda m: EXTRACTION_METHODS[m],
    help="AI extraction appears when OPENAI_API_KEY is set in .env (restart the app).",
)
if method == "llm":
    st.warning(
        f"AI extraction sends the document text to OpenAI (model {lab.settings.openai_model}) "
        "and is billed to your account."
    )

uploads = st.file_uploader(
    f"Upload documents (max {lab.settings.max_upload_mb} MB each)",
    type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS],
    accept_multiple_files=True,
)
if uploads and st.button("Process uploads", type="primary"):
    for upload in uploads:
        with st.spinner(f"Processing {upload.name}…"):
            try:
                outcome = lab.ingest(upload.name, upload.getvalue(), method)
            except UploadValidationError as exc:
                st.error(str(exc))
                continue
        if outcome.status == "processed":
            st.success(
                f"{outcome.filename}: {outcome.entities} entities, "
                f"{outcome.relationships} relationship statements."
            )
        elif outcome.status == "skipped":
            st.info(f"{upload.name}: skipped. {outcome.message}")
        else:
            st.error(f"{outcome.filename}: {outcome.message}")

st.subheader("Uploaded documents")
documents = lab.registry.list_documents()
table(
    (
        {
            "Id": d.id,
            "File": d.original_filename,
            "Type": d.file_type,
            "Size (bytes)": d.file_size,
            "Status": d.processing_status,
            "Processed (UTC)": d.processed_at.strftime("%Y-%m-%d %H:%M") if d.processed_at else "",
            "SHA-256": d.file_hash[:12] + "…",
            "Error": d.error_message or "",
        }
        for d in documents
    ),
    empty="No documents yet.",
)
if not documents:
    st.stop()

selected = st.selectbox(
    "Inspect a document", documents, format_func=lambda d: f"{d.id} · {d.original_filename}"
)

reprocess, remove, confirm = st.columns([1, 1, 3])
if reprocess.button("Reprocess"):
    with st.spinner("Reprocessing…"):
        outcome = lab.process(selected.id, method)
    if outcome.status == "failed":
        st.error(outcome.message)
    else:
        st.success("Reprocessed. Graph, vectors and provenance are up to date.")
confirmed = confirm.checkbox(
    "Yes, delete this document and its derived data", key=f"confirm-{selected.id}"
)
if remove.button("Delete", disabled=not confirmed):
    lab.delete_document(selected.id)
    st.rerun()

document = lab.registry.get(selected.id)
if document.error_message:
    st.error(f"Last processing error: {document.error_message}")

extraction = lab.extraction(document.id)
tabs = st.tabs(["Relationships", "Entities", "Events & triplets", "Discarded", "Parsed text", "Runs"])
if extraction is None:
    tabs[0].caption("No successful extraction for this document.")
else:
    with tabs[0]:
        table(
            {
                "Subject": f.subject,
                "Predicate": f.predicate,
                "Object": f.object,
                "Location": f.location,
                "Confidence": f.confidence,
                "Valid from": f.valid_from or "",
                "Valid until": f.valid_until or "",
                "Statement": f.quote,
            }
            for f in extraction.facts
        )
    with tabs[1]:
        grouped: dict[tuple[str, str], int] = {}
        for mention in extraction.mentions:
            key = (mention.entity_type, mention.text)
            grouped[key] = grouped.get(key, 0) + 1
        table(
            {"Type": kind, "Surface form": text, "Mentions": count}
            for (kind, text), count in sorted(grouped.items())
        )
    with tabs[2]:
        st.caption(
            "Exploratory output of Semantica's pattern EventDetector. Its participant lists are "
            "noisy, so events are not written to the graph."
        )
        table(extraction.events, empty="No events detected.")
        st.markdown("**Triplets** (Semantica `TripletExtractor`)")
        table(
            {"Subject": s, "Predicate": p, "Object": o} for s, p, o in extraction.triplets
        )
    with tabs[3]:
        st.caption("Candidates produced by Semantica that were kept out of the graph, and why.")
        table(
            ({"Reason": reason, "Count": count} for reason, count in extraction.discarded.items()),
            empty="Nothing was discarded.",
        )

with tabs[4]:
    path = lab.settings.exports_dir / f"parsed_{document.id}.json"
    if path.exists():
        parsed = json.loads(path.read_text(encoding="utf-8"))
        st.caption(f"Parser: {parsed['parser']} · segments: {len(parsed['segments'])}")
        if parsed["metadata"]:
            st.json(parsed["metadata"], expanded=False)
        st.text_area("Normalised text", parsed["text"], height=320, disabled=True)
    else:
        st.caption("Not parsed yet.")
with tabs[5]:
    table(
        {
            "Run": r.id,
            "Status": r.status,
            "Method": r.extraction_method,
            "Entities": r.entities_count,
            "Relationships": r.relationships_count,
            "Seconds": round(r.processing_duration or 0, 2),
            "Error": r.error_message or "",
        }
        for r in reversed(document.runs)
    )
