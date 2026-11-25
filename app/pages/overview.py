"""Page 1: totals and recent processing activity."""

import streamlit as st

from app.components.common import get_lab, require_graph, require_vectors, table

lab = get_lab()
st.title("Overview")

graph = require_graph(lab)
analysis = lab.analysis()
counts = lab.registry.counts()

first = st.columns(4)
first[0].metric("Documents", counts["documents"])
first[1].metric("Entities", len(graph["entities"]))
first[2].metric("Relationships", len(graph["relationships"]))
first[3].metric("Ingestion runs", counts["ingestion_runs"])
second = st.columns(4)
second[0].metric("Detected conflicts", len(analysis["conflicts"]))
second[1].metric("Merged entity groups", len(analysis["merges"]))
second[2].metric("Duplicates to review", len(analysis["duplicate_candidates"]))
second[3].metric("Vector chunks", require_vectors(lab))

if not counts["documents"]:
    st.info("The knowledge base is empty. Load the fictional Acme AI corpus, or upload files.")
    if st.button("Load sample documents", type="primary"):
        with st.spinner("Running the pipeline on six documents…"):
            outcomes = lab.load_sample_data()
        failed = [o for o in outcomes if o.status == "failed"]
        for outcome in failed:
            st.error(f"{outcome.filename}: {outcome.message}")
        if not failed:
            st.rerun()

if graph["entities"]:
    left, right = st.columns(2)
    for column, title, key, items in (
        (left, "Entities by type", "Entity type", graph["entities"]),
        (right, "Relationships by type", "Relationship type", graph["relationships"]),
    ):
        totals: dict[str, int] = {}
        for item in items:
            totals[item["type"]] = totals.get(item["type"], 0) + 1
        with column:
            st.subheader(title)
            table(
                {key: name, "Count": count}
                for name, count in sorted(totals.items(), key=lambda t: -t[1])
            )

st.subheader("Recent processing activity")
table(
    (
        {
            "Run": run.id,
            "Document": run.document.original_filename,
            "Status": run.status,
            "Method": run.extraction_method,
            "Entities": run.entities_count,
            "Relationships": run.relationships_count,
            "Seconds": round(run.processing_duration or 0, 2),
            "Started (UTC)": run.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "Error": run.error_message or "",
        }
        for run in lab.registry.recent_runs()
    ),
    empty="No documents have been processed.",
)

if sync := analysis.get("last_sync"):
    st.caption(
        "Last Neo4j synchronisation: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sync.items())
    )
