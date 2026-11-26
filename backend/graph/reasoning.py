"""Rule inference and point-in-time views (Semantica ``Reasoner`` and temporal query).

Inferred facts are returned for display only.  They are never written to the
graph, so extracted (sourced) knowledge and derived knowledge stay distinct.
Relationships whose evidence status is ``inferred`` are not used as premises.
"""

from __future__ import annotations

from semantica.kg import TemporalGraphQuery
from semantica.reasoning import Reasoner

RULES: tuple[tuple[str, str], ...] = (
    (
        "Indirect dependency",
        "IF DEPENDS_ON(?a, ?b) AND DEPENDS_ON(?b, ?c) THEN DEPENDS_ON_INDIRECTLY(?a, ?c)",
    ),
    (
        "Accountable person",
        "IF LEADS(?p, ?t) AND OWNS(?t, ?x) THEN ACCOUNTABLE_FOR(?p, ?x)",
    ),
    (
        "Affiliation through a team",
        "IF PART_OF(?p, ?t) AND PART_OF(?t, ?o) THEN AFFILIATED_WITH(?p, ?o)",
    ),
    (
        "Team technology exposure",
        "IF OWNS(?t, ?x) AND DEPENDS_ON(?x, ?y) THEN TEAM_RELIES_ON(?t, ?y)",
    ),
)


def infer(graph: dict[str, list[dict]], rules: tuple[tuple[str, str], ...] = RULES) -> list[dict]:
    """Forward-chain the rules over the directly supported relationships."""
    names = {entity["id"]: entity["name"] for entity in graph["entities"]}
    reasoner = Reasoner()
    labels: dict[str, str] = {}
    for label, rule in rules:
        added = reasoner.add_rule(rule)
        labels[added.rule_id] = label
    for relationship in graph["relationships"]:
        if relationship.get("support_status") == "inferred":
            continue
        reasoner.add_fact(
            f"{relationship['type']}({names[relationship['source']]}, "
            f"{names[relationship['target']]})"
        )
    results = []
    for result in reasoner.forward_chain():
        rule = result.rule_used
        results.append(
            {
                "conclusion": str(result.conclusion),
                "rule": labels.get(getattr(rule, "rule_id", ""), getattr(rule, "rule_id", "")),
                "premises": [str(premise) for premise in result.premises],
            }
        )
    return sorted(results, key=lambda r: (r["rule"], r["conclusion"]))


def point_in_time(graph: dict[str, list[dict]], date: str) -> dict[str, list[dict]]:
    """Relationships valid on ``date`` (undated facts are always valid)."""
    snapshot = TemporalGraphQuery().query_at_time(graph, "", at_time=date)
    return {"entities": snapshot["entities"], "relationships": snapshot["relationships"]}
