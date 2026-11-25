"""Interactive graph figures from Semantica's ``KGVisualizer`` (Plotly)."""

from __future__ import annotations

from pathlib import Path

from semantica.visualization import KGVisualizer


def neighbourhood(graph: dict[str, list[dict]], centre: str, depth: int) -> dict[str, list[dict]]:
    """The subgraph within ``depth`` hops of ``centre`` (for focused views)."""
    keep = {centre}
    for _ in range(depth):
        keep |= {
            other
            for rel in graph["relationships"]
            if rel["source"] in keep or rel["target"] in keep
            for other in (rel["source"], rel["target"])
        }
    return {
        "entities": [e for e in graph["entities"] if e["id"] in keep],
        "relationships": [
            r for r in graph["relationships"] if r["source"] in keep and r["target"] in keep
        ],
    }


def network_figure(graph: dict[str, list[dict]], layout: str = "force"):
    """A Plotly figure with one marker per entity, coloured by type."""
    figure = KGVisualizer(layout=layout).visualize_network(
        graph, output="interactive", node_color_by="type"
    )
    figure.update_layout(height=680, margin={"l": 10, "r": 10, "t": 40, "b": 10})
    # The last trace holds the entity markers; larger ones are easier to click.
    figure.data[-1].marker.size = 18
    return figure


def export_html(graph: dict[str, list[dict]], path: Path) -> Path:
    """Standalone HTML artifact produced by Semantica."""
    path.parent.mkdir(parents=True, exist_ok=True)
    KGVisualizer(layout="force").visualize_network(graph, output="html", file_path=path)
    return path
