"""Page 5: facts and their sources, conflicts, duplicates, integrity and decisions."""

import streamlit as st

from app.components.common import entity_label, get_lab, require_graph, table
from backend.graph.conflicts import DEFAULT_CREDIBILITY, detect_conflicts
from backend.services.decisions import SAMPLE_DECISION, DecisionService

@st.cache_resource(show_spinner="Preparing decision tracking…")
def decision_service(_lab) -> DecisionService:
    return DecisionService(_lab)


lab = get_lab()
st.title("Provenance & Conflicts")
graph = require_graph(lab)
analysis = lab.analysis()
documents = lab.registry.list_documents()

facts_tab, conflicts_tab, duplicates_tab, integrity_tab, decisions_tab = st.tabs(
    ["Facts", "Conflicts", "Duplicate entities", "Integrity & export", "Decision trace"]
)

with facts_tab:
    facts = [f for d in documents if (e := lab.extraction(d.id)) for f in
             [{"document": e.filename, **fact.model_dump()} for fact in e.facts]]
    filters = st.columns(3)
    by_document = filters[0].multiselect("Document", sorted({f["document"] for f in facts}))
    by_predicate = filters[1].multiselect("Predicate", sorted({f["predicate"] for f in facts}))
    needle = filters[2].text_input("Text contains")
    shown = [
        f
        for f in facts
        if (not by_document or f["document"] in by_document)
        and (not by_predicate or f["predicate"] in by_predicate)
        and needle.lower() in f"{f['subject']} {f['object']} {f['quote']}".lower()
    ]
    st.caption(f"{len(shown)} of {len(facts)} extracted facts")
    table(
        (
            {
                "Subject": f["subject"],
                "Predicate": f["predicate"],
                "Object": f["object"],
                "Source": f["document"],
                "Location": f["location"],
                "Confidence": f["confidence"],
                "Method": f["method"],
                "Extracted": f["extracted_at"][:19],
                "Original statement": f["quote"],
            }
            for f in shown
        ),
        empty="No facts match.",
    )
    st.caption(
        "Confidence is whatever the extractor reports. The pattern and regex methods use fixed "
        "constants, so it is not a calibrated probability."
    )
    if shown:
        picked = st.selectbox(
            "Trace one fact",
            shown,
            format_func=lambda f: f"{f['subject']} {f['predicate']} {f['object']} ({f['document']})",
        )
        lineage = lab.provenance.lineage(picked["fact_id"])
        if not lineage:
            st.warning("No provenance record found.")
        else:
            st.write(
                f"Sources: {', '.join(lineage['source_documents'])} · checksum verified: "
                f"{lineage['integrity_verified']}"
            )
            table(
                {
                    "Record": entry["entity_id"],
                    "Kind": entry["entity_type"],
                    "Source": entry["source_document"],
                    "Location": entry["source_location"] or "",
                    "Activity": entry["activity_id"],
                    "Derived from": entry["parent_entity_id"] or "",
                    "Recorded": entry["timestamp"][:19],
                }
                for entry in lineage["lineage_chain"]
            )

with conflicts_tab:
    st.caption(
        "Detected by Semantica `ConflictDetector`. Both claims are kept; nothing is overwritten."
    )
    credibility = {}
    with st.expander("Source credibility (only affects the suggestion)"):
        for document in sorted({o["source"] for o in analysis["observations"]}):
            credibility[document] = st.slider(
                document, 0.0, 1.0, DEFAULT_CREDIBILITY, 0.05, key=f"cred-{document}"
            )
    conflicts, successions = detect_conflicts(analysis["observations"], credibility)
    if not conflicts:
        st.success("No conflicting claims detected.")
    for conflict in conflicts:
        with st.container(border=True):
            st.markdown(
                f"⚠️ **{conflict['entity_name']} · `{conflict['property']}`**: "
                f"{' vs '.join(conflict['values'])}  (severity {conflict['severity']})"
            )
            table(
                {
                    "Claim": c["value"],
                    "Source": c["document"],
                    "Location": c["location"],
                    "Confidence": c["confidence"],
                    "Statement": c["quote"],
                }
                for c in conflict["claims"]
            )
            st.caption(f"Recommended action: {conflict['recommended_action']}")
            for suggestion in conflict["suggestions"]:
                if suggestion["tie"]:
                    st.caption(
                        f"`{suggestion['strategy']}`: **tie**. The evidence does not favour "
                        "either value; a person has to decide."
                    )
                else:
                    st.caption(
                        f"Suggestion by `{suggestion['strategy']}` (not applied): "
                        f"**{suggestion['value']}** — {suggestion['notes']}"
                    )
    if successions:
        st.subheader("Changes over time (not conflicts)")
        for succession in successions:
            st.markdown(f"**{succession['entity_name']} · `{succession['property']}`**")
            table(
                {
                    "Value": c["value"],
                    "Valid from": c["valid_from"] or "",
                    "Valid until": c["valid_until"] or "",
                    "Source": c["document"],
                    "Statement": c["quote"],
                }
                for c in succession["claims"]
            )

with duplicates_tab:
    st.subheader("Merged")
    table(
        (
            {
                "Canonical": m["canonical"],
                "Type": m["type"],
                "Surface forms": ", ".join(m["merged"]),
                "Why": m["method"],
                "Score": m["score"],
            }
            for m in analysis["merges"]
        ),
        empty="No surface forms needed merging.",
    )
    st.subheader("Candidates for review (kept separate)")
    st.caption(
        "Pairs Semantica `DuplicateDetector` scores as similar but below the auto-merge threshold. "
        "Sibling names such as two 'Project …' entities score high, which is why similarity "
        "alone does not merge them."
    )
    table(
        (
            {
                "First": r["first"],
                "Second": r["second"],
                "Type": r["type"],
                "Similarity": r["similarity"],
                "Confidence": r["confidence"],
                "Reasons": ", ".join(r["reasons"]),
            }
            for r in analysis["duplicate_candidates"]
        ),
        empty="No candidates.",
    )

with integrity_tab:
    st.json(lab.provenance.statistics())
    if st.button("Verify provenance hash chain"):
        outcome = lab.provenance.verify_integrity()
        (st.success if outcome["valid"] else st.error)(
            f"Valid: {outcome['valid']} · entries: {outcome['total_entries']} · "
            f"broken links: {len(outcome['broken_links'])}"
        )
    if st.button("Export PROV-O (Turtle)"):
        st.download_button(
            "Download provenance.ttl",
            lab.provenance.export_prov_turtle(),
            file_name="provenance.ttl",
            mime="text/turtle",
        )

with decisions_tab:
    st.caption(
        "Optional experiment with Semantica `AgentContext` decision tracking. Decisions are "
        "stored in Neo4j and linked to the entities they concern."
    )
    entities = sorted(graph["entities"], key=lambda e: e["name"].lower())
    if not entities:
        st.info("Load documents first.")
        st.stop()
    with st.form("decision"):
        category = st.text_input("Category", SAMPLE_DECISION["category"])
        scenario = st.text_input("Scenario", SAMPLE_DECISION["scenario"])
        outcome_text = st.text_input("Outcome", SAMPLE_DECISION["outcome"])
        reasoning = st.text_area("Reasoning", SAMPLE_DECISION["reasoning"])
        confidence = st.slider("Confidence", 0.0, 1.0, SAMPLE_DECISION["confidence"], 0.05)
        linked = st.multiselect(
            "Supporting entities",
            entities,
            default=[e for e in entities if e["id"] in SAMPLE_DECISION["entities"]],
            format_func=entity_label,
        )
        submitted = st.form_submit_button("Record decision")
    if submitted:
        if not (category.strip() and scenario.strip() and outcome_text.strip() and linked):
            st.error("Category, scenario, outcome and at least one entity are required.")
        else:
            with st.spinner("Recording…"):
                identifier = decision_service(lab).record(
                    category=category,
                    scenario=scenario,
                    reasoning=reasoning,
                    outcome=outcome_text,
                    confidence=confidence,
                    entities=[e["id"] for e in linked],
                )
            st.success(f"Recorded decision {identifier}")

    if st.toggle("Show recorded decisions"):
        service = decision_service(lab)
        recorded = service.list_decisions()
        table(recorded, empty="No decisions recorded.")
        if recorded:
            chosen = st.selectbox(
                "Trace a decision", recorded, format_func=lambda d: f"{d['scenario']} → {d['outcome']}"
            )
            trace = service.trace(chosen["decision_id"])
            st.write("Linked entities:", ", ".join(trace["linked_entities"]))
            st.write("Source documents:", ", ".join(trace["source_documents"]) or "none")
            table(trace["supporting_facts"], empty="No facts connect the linked entities.")
            for conflict in trace["open_conflicts"]:
                st.warning(
                    f"Open conflict on {conflict['entity_name']}.{conflict['property']}: "
                    f"{' vs '.join(conflict['values'])}"
                )
            with st.expander("Semantica trace_decision_explainability output"):
                st.json(trace["semantica_trace"], expanded=False)
            st.caption(service.probe_precedent_search(chosen["scenario"]))
