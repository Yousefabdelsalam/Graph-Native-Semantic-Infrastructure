"""Graph analytics with Semantica's ``CentralityCalculator`` and ``ConnectivityAnalyzer``."""

from __future__ import annotations

from semantica.kg import CentralityCalculator, ConnectivityAnalyzer


def analyse(graph: dict[str, list[dict]], top: int = 10) -> dict:
    """Most central entities and the connected components of the graph."""
    calculator = CentralityCalculator()
    measures = {
        "Degree": calculator.calculate_degree_centrality,
        "Betweenness": calculator.calculate_betweenness_centrality,
        "PageRank": calculator.calculate_pagerank,
    }
    names = {entity["id"]: entity["name"] for entity in graph["entities"]}
    rankings: dict[str, list[dict]] = {}
    for label, measure in measures.items():
        # 0.7.0 returns dicts for degree and betweenness but tuples for PageRank.
        pairs = (
            (row["node"], row["score"]) if isinstance(row, dict) else row
            for row in measure(graph)["rankings"][:top]
        )
        rankings[label] = [
            {"Entity": names.get(node, node), "Score": round(float(score), 4)}
            for node, score in pairs
        ]
    connectivity = ConnectivityAnalyzer().analyze_connectivity(graph)
    return {
        "rankings": rankings,
        "components": [
            sorted(names.get(node, node) for node in component)
            for component in sorted(connectivity["components"], key=len, reverse=True)
        ],
        "density": round(float(connectivity["density"]), 4),
        "average_degree": round(float(connectivity["avg_degree"]), 2),
    }
