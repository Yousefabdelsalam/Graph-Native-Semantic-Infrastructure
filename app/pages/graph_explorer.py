"""Page 3: interactive graph, entity inspection, ontology and reasoning."""

from datetime import date

import streamlit as st

from app.components.common import entity_label, get_lab, require_graph, table
from backend.graph.analytics import analyse
from backend.graph.reasoning import RULES, infer, point_in_time
from backend.graph.visualize import export_html, neighbourhood, network_figure
from backend.ontology.schema import ENTITY_TYPES
from backend.ontology.service import OntologyService

lab = get_lab()
st.title("Knowledge Graph Explorer")
graph = require_graph(lab)
if not graph["entities"]:
    st.info("The graph is empty. Load or upload documents first.")
    st.stop()

by_id = {entity["id"]: entity for entity in graph["entities"]}
ordered = sorted(graph["entities"], key=lambda e: (e["name"].lower(), e["type"]))

explore, analytics_tab, ontology_tab, reasoning_tab = st.tabs(
    ["Explore", "Analytics", "Ontology & validation", "Reasoning & time"]
)

with explore:
    controls = st.columns([3, 1, 3, 1])
    choice = controls[0].selectbox(
        "Search an entity",
        [None, *ordered],
        format_func=lambda e: "(whole graph)" if e is None else entity_label(e),
    )
    depth = controls[1].number_input("Hops", 1, 4, 1, disabled=choice is None)
    all_types = sorted({r["type"] for r in graph["relationships"]})
    wanted = controls[2].multiselect("Relationship types", all_types, placeholder="All types")
    edge_labels = controls[3].toggle("Edge labels", value=choice is not None)

    view = graph
    if wanted:
        view = {**view, "relationships": [r for r in view["relationships"] if r["type"] in wanted]}
    if choice is not None:
        view = neighbourhood(view, choice["id"], int(depth))

    figure = network_figure(view)
    if not edge_labels and len(figure.data) == 3:
        figure.data[1].visible = False
    colours = dict(zip((e["type"] for e in view["entities"]), figure.data[-1].marker.color))
    st.markdown(
        " &nbsp; ".join(
            f"<span style='color:{colour}'>●</span> {kind}"
            for kind, colour in sorted(colours.items())
            if kind in ENTITY_TYPES
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        f"{len(view['entities'])} entities, {len(view['relationships'])} relationships. "
        "Drag to zoom, hover for type, click a node to inspect it."
    )
    event = st.plotly_chart(figure, on_select="rerun", selection_mode="points", key="graph")

    clicked = None
    for point in (event.selection.points if event else []):
        # The last trace holds the nodes, in the order of view["entities"].
        if point.get("curve_number") == len(figure.data) - 1:
            index = point.get("point_index")
            if index is not None and index < len(view["entities"]):
                clicked = view["entities"][index]
    entity = clicked or choice

    if entity is None:
        st.info("Select an entity above, or click a node, to see its properties and sources.")
    else:
        st.subheader(entity_label(entity))
        left, right = st.columns([1, 2])
        with left:
            st.markdown("**Properties** (as stored in Neo4j)")
            st.json(entity)
        connected = [
            r for r in graph["relationships"] if entity["id"] in (r["source"], r["target"])
        ]
        with right:
            st.markdown("**Connections**")
            table(
                (
                    {
                        "Source": by_id[r["source"]]["name"],
                        "Relationship": r["type"],
                        "Target": by_id[r["target"]]["name"],
                        "Confidence": r.get("confidence"),
                        "Documents": ", ".join(r.get("source_documents", [])),
                    }
                    for r in connected
                ),
                empty="This entity has no relationships.",
            )
            neighbours = lab.repository.neighbors(entity["id"], depth=2)
            st.caption(
                "Within two hops (Semantica `GraphStore.get_neighbors`): "
                + (", ".join(sorted(n["name"] for n in neighbours)) or "none")
            )

        if connected:
            st.markdown("**Provenance of a relationship**")
            relationship = st.selectbox(
                "Relationship",
                connected,
                format_func=lambda r: f"{by_id[r['source']]['name']} {r['type']} {by_id[r['target']]['name']}",
            )
            for fact_id in relationship.get("provenance_ids", []):
                record = lab.provenance.get(fact_id) or {}
                metadata = record.get("metadata") or {}
                with st.container(border=True):
                    # Plain text: quotes come from uploaded documents.
                    st.text(record.get("source_quote") or "(no quote recorded)")
                    st.caption(
                        f"**{record.get('source_document')}** · {record.get('source_location')} · "
                        f"method `{metadata.get('extraction_method')}` · confidence "
                        f"{record.get('confidence')} · extracted {metadata.get('extracted_at')}"
                    )
                    with st.expander("PROV lineage (Semantica ProvenanceManager)"):
                        st.json(lab.provenance.lineage(fact_id), expanded=False)

    with st.expander("Connect two entities"):
        pair = st.columns(2)
        start = pair[0].selectbox("From", ordered, format_func=entity_label, key="from")
        end = pair[1].selectbox(
            "To", ordered, index=len(ordered) - 1, format_func=entity_label, key="to"
        )
        if st.button("Find shortest path"):
            path = lab.repository.shortest_path(start["id"], end["id"], max_depth=6)
            if path is None:
                st.warning("No path of up to six hops.")
            else:
                st.success(
                    f"{path['length']} hop(s): "
                    + " → ".join(by_id[i]["name"] for i in path["node_ids"])
                )
                edges = {r["id"]: r for r in graph["relationships"]}
                table(
                    {
                        "Source": by_id[edges[i]["source"]]["name"],
                        "Relationship": edges[i]["type"],
                        "Target": by_id[edges[i]["target"]]["name"],
                        "Documents": ", ".join(edges[i]["source_documents"]),
                    }
                    for i in path["relationship_ids"]
                )

    if st.button("Export this view as standalone HTML"):
        target = export_html(view, lab.settings.exports_dir / "knowledge_graph.html")
        st.download_button(
            "Download HTML", target.read_bytes(), file_name=target.name, mime="text/html"
        )

with analytics_tab:
    st.markdown("**Centrality and connectivity** (Semantica `CentralityCalculator`, `ConnectivityAnalyzer`)")
    if st.button("Analyse graph"):
        result = analyse(graph)
        for column, (measure, rows) in zip(st.columns(3), result["rankings"].items()):
            column.markdown(f"**{measure}**")
            column.dataframe(rows, hide_index=True)
        st.caption(
            f"Density {result['density']} · average degree {result['average_degree']} · "
            f"{len(result['components'])} connected component(s)"
        )
        table(
            {"Component": number, "Size": len(members), "Entities": ", ".join(members)}
            for number, members in enumerate(result["components"], start=1)
        )

with ontology_tab:
    service = OntologyService()
    st.markdown(
        "The vocabulary is declared by this app; Semantica renders it to **OWL** and **SHACL** "
        "and validates the exported graph."
    )
    if st.button("Validate graph against SHACL"):
        for title, shapes in (
            ("Shapes exactly as Semantica 0.7.0 generates them", service.shacl()),
            ("Shapes with all allowed ranges (`sh:or`, application correction)", service.corrected_shacl()),
        ):
            report = service.validate(graph, shapes)
            st.markdown(f"**{title}**")
            if report["conforms"]:
                st.success("Conforms.")
            else:
                st.error(f"{len(report['violations'])} violation(s)")
                table(
                    {
                        "Node": v["focus_node"].split("#")[-1],
                        "Property": (v.get("result_path") or "").split("#")[-1],
                        "Constraint": v["constraint"],
                        "Value": (v.get("value") or "").split("#")[-1],
                    }
                    for v in report["violations"]
                )
        st.caption(
            "Violations that disappear in the second table are false positives caused by the "
            "generator keeping only the first range of a multi-range property."
        )
    with st.expander("OWL (Turtle)"):
        st.code(service.owl(), language="turtle")
    with st.expander("SHACL shapes as generated (Turtle)"):
        st.code(service.shacl(), language="turtle")
    with st.expander("Ontology inferred from the data by Semantica, for comparison"):
        if st.button("Infer ontology"):
            inferred = service.infer_from_graph(graph)
            st.write("Classes:", ", ".join(c["name"] for c in inferred["classes"]))
            st.write("Properties:", ", ".join(p["name"] for p in inferred["properties"]))
            st.caption(
                "Classes seen only once are omitted and internal storage fields appear as "
                "data properties, which is why the declared vocabulary is used for validation."
            )

with reasoning_tab:
    st.markdown("**Forward-chaining inference** (Semantica `Reasoner`). Results are not stored.")
    for label, rule in RULES:
        st.code(f"{label}: {rule}", language="text")
    if st.button("Run inference"):
        table(
            (
                {"Inferred": i["conclusion"], "Rule": i["rule"], "Because": " & ".join(i["premises"])}
                for i in infer(graph)
            ),
            empty="No new facts.",
        )
    st.divider()
    st.markdown("**Point-in-time view** (Semantica `TemporalGraphQuery`)")
    moment = st.date_input(
        "Facts valid on", date(2024, 6, 30), min_value=date(2000, 1, 1), max_value=date(2100, 1, 1)
    )
    snapshot = point_in_time(graph, moment.isoformat())
    timed = [r for r in graph["relationships"] if r.get("valid_from") or r.get("valid_until")]
    active = {r["id"] for r in snapshot["relationships"]}
    st.caption(
        f"{len(snapshot['relationships'])} of {len(graph['relationships'])} relationships are "
        "valid on that date. Undated facts always count as valid."
    )
    table(
        (
            {
                "Source": by_id[r["source"]]["name"],
                "Relationship": r["type"],
                "Target": by_id[r["target"]]["name"],
                "Valid from": r.get("valid_from") or "",
                "Valid until": r.get("valid_until") or "",
                "Active on date": r["id"] in active,
            }
            for r in timed
        ),
        empty="No time-bounded facts in the graph.",
    )
