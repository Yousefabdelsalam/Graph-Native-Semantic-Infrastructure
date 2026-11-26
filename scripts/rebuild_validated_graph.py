"""Rebuild the graph from stored extraction results, with evidence validation.

No document is re-extracted and no model is called.  The uploaded files and the
stored extraction results are checked against the checksums taken at backup time,
before and after the rebuild.  The report is written to
``data/exports/quality/report.md`` (ignored by git: it quotes your documents).

    python scripts/rebuild_validated_graph.py --backup "<backup folder>"

The backup folder holds ``data/`` (a copy made before this run) and
``checksums_data_before.txt`` (``sha256sum`` lines relative to ``data/``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import SecretStr  # noqa: E402

from backend.config import configure_logging, load_settings  # noqa: E402
from backend.extraction.validation import RULES_VERSION, relationship_flags  # noqa: E402
from backend.llm.provider import MeteredOpenAIProvider, usage_meter  # noqa: E402
from backend.services.lab import KnowledgeLab  # noqa: E402

_ORIGINALS = ("./uploads/", "./exports/extractions/", "./exports/parsed_")


def _refuse_model_calls(*_args, **_kwargs):
    raise RuntimeError("This rebuild must not call a model.")


def checksums(backup: Path) -> dict[str, str]:
    """Recorded SHA-256 of every original file (uploads and raw results)."""
    recorded = {}
    for line in (backup / "checksums_data_before.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if name.startswith(_ORIGINALS):
            recorded[name] = digest
    if not recorded:
        raise SystemExit("The backup holds no checksums of original files.")
    return recorded


def verify_originals(data_dir: Path, recorded: dict[str, str]) -> list[tuple[str, str, bool]]:
    rows = []
    for name, digest in sorted(recorded.items()):
        path = data_dir / name
        current = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
        rows.append((name[2:], digest, current == digest))
    return rows


def _name(entity: dict) -> str:
    return f"{entity['name']} [{entity['type']}]"


def _quote(text: str) -> str:
    return "> " + " ".join(text.split())


def build_report(lab, assembled, before: dict, context: dict) -> str:
    after = lab.graph()
    analysis = lab.analysis()
    documents = {d.id: d.original_filename for d in lab.registry.list_documents()}
    entities = {e["id"]: e for e in after["entities"]}
    resolve = assembled.resolution.resolve

    reports = {
        int(path.stem): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(lab.settings.validation_dir.glob("*.json"))
    }
    extractions = {i: lab.extraction(i) for i in documents}
    counts = Counter(v["status"] for r in reports.values() for v in r["verdicts"])
    edge_status = Counter(r["support_status"] for r in after["relationships"])
    edge_of = {fact: r for r in after["relationships"] for fact in r["provenance_ids"]}
    folded = {f["fact_id"]: f for f in analysis["folded_relationships"]}

    # Entities: where did every previous node go?
    moved: dict[str, list[dict]] = defaultdict(list)
    removed = []
    for old in before["entities"]:
        new_id = resolve(old["type"], old["name"])
        for alias in old.get("aliases", []):
            new_id = new_id or resolve(old["type"], alias)
        if new_id is None:
            removed.append(old)
        else:
            moved[new_id].append(old)
    merged = {new: olds for new, olds in moved.items() if len(olds) > 1}
    added = [e for i, e in entities.items() if i not in moved]

    out = [
        "# Relationship quality report",
        "",
        f"Generated {context['generated']} · validation rules version {RULES_VERSION} · "
        f"model requests during this rebuild: **{context['model_requests']}**",
        "",
        "## Summary",
        "",
        "| Measure | Before | After |",
        "|---|---|---|",
        f"| Entities | {len(before['entities'])} | {len(after['entities'])} |",
        f"| Relationships in the graph | {len(before['relationships'])} | "
        f"{len(after['relationships'])} |",
        f"| of which supported | not recorded | {edge_status['supported']} |",
        f"| of which inferred | not recorded | {edge_status['inferred']} |",
        "",
        f"Extracted relationship statements checked: **{sum(counts.values())}**: "
        f"**{counts['supported']} supported**, **{counts['inferred']} inferred**, "
        f"**{counts['rejected']} rejected**. Relationship statements merged into another "
        f"relationship: **{sum(len(r['provenance_ids']) - 1 for r in after['relationships'])}**. "
        f"Entity nodes combined: **{sum(len(o) for o in merged.values())} into "
        f"{len(merged)}**. Entity nodes removed: **{len(removed)}**.",
        "",
        "- **Supported**: both names are in one sentence of body text or one table row. The "
        "rules do not judge whether the sentence expresses the verb correctly.",
        "- **Inferred**: follows from a section heading and its content. **Not a confirmed "
        "fact.**",
        "- **Rejected**: not in the graph. The raw result is kept and the provenance entry is "
        "marked invalid, with the reason.",
        "",
    ]

    out += [
        "## Overview",
        "",
        "| # | Relationship statement | Status | Rule |",
        "|---|---|---|---|",
    ]
    out += [
        f"| {number} | {v['subject']} {v['predicate']} {v['object']} | {v['status']} | {v['rule']} |"
        for report in reports.values()
        for number, v in enumerate(report["verdicts"], start=1)
    ]
    out.append("")

    for status in ("supported", "inferred", "rejected"):
        out += [f"## {status.capitalize()} relationships", ""]
        found = False
        for document_id, report in reports.items():
            for number, verdict in enumerate(report["verdicts"], start=1):
                if verdict["status"] != status:
                    continue
                found = True
                ends = []
                for side in ("subject", "object"):
                    new_id = resolve(verdict[f"{side}_type"], verdict[side])
                    node = entities.get(new_id)
                    if node and node["name"] != verdict[side]:
                        ends.append(f"'{verdict[side]}' is shown as node '{node['name']}'")
                out += [
                    f"### #{number} · {verdict['subject']} — {verdict['predicate']} → "
                    f"{verdict['object']}",
                    "",
                    f"- Document: {report['filename']} (id {document_id})",
                    f"- Rule: {verdict['rule']}",
                    f"- Reason: {verdict['reason']}",
                    f"- Model confidence (not used by the rules): {verdict['confidence']}",
                ]
                if status == "rejected":
                    out += [
                        f"- Stored location: {verdict['original_location']}",
                        "- Evidence as originally extracted (not accepted):",
                        "",
                        _quote(verdict["original_quote"]),
                        "",
                    ]
                    continue
                if ends:
                    out.append(f"- Entity merge: {'; '.join(ends)}")
                if verdict["fact_id"] in folded:
                    out.append("- Merged into: " + " ".join(folded[verdict["fact_id"]]["into"]))
                edge = edge_of.get(verdict["fact_id"])
                if edge is None:
                    out.append("- **Not in the graph** (an end could not be resolved).")
                for flag in (edge or {}).get("review_flags", []):
                    out.append(f"- Review: {flag}")
                out += [
                    f"- Location (cleaned text): {verdict['location']}",
                    "- Evidence:",
                    "",
                    _quote(verdict["quote"]),
                    "",
                ]
                if " ".join(verdict["original_quote"].split()) != " ".join(verdict["quote"].split()):
                    out += [
                        "- Evidence as originally extracted:",
                        "",
                        _quote(verdict["original_quote"]),
                        "",
                    ]
        if not found:
            out += ["None.", ""]

    out += ["## Merged relationships", ""]
    combined = [r for r in after["relationships"] if len(r["provenance_ids"]) > 1]
    if not combined:
        out += [
            "None. No two accepted statements ended up as the same relationship after entity "
            "merging, and no 'related to' link duplicated a more specific one.",
            "",
        ]
    for r in combined:
        out.append(
            f"- {entities[r['source']]['name']} {r['type']} {entities[r['target']]['name']} "
            f"({r['support_status']}): {len(r['provenance_ids'])} statements"
        )
        for quote, status in zip(r["evidence_quotes"], r["evidence_support"]):
            out += ["", _quote(f"[{status}] {quote}")]
        out.append("")

    out += ["## Reversed duplicates and conflicts", ""]
    candidates = []
    labels = {}
    for report in reports.values():
        for number, v in enumerate(report["verdicts"], start=1):
            source = resolve(v["subject_type"], v["subject"]) or v["subject"].lower()
            target = resolve(v["object_type"], v["object"]) or v["object"].lower()
            candidates.append((v["fact_id"], source, v["predicate"], target))
            labels[v["fact_id"]] = (
                f"#{number} {v['subject']} {v['predicate']} {v['object']} ({v['status']})"
            )
    flags = relationship_flags(candidates)
    out.append(
        f"Flagged in the graph: **{len(analysis['relationship_flags'])}**. Checked across all "
        f"extracted statements, including rejected ones: **{len(flags)}**."
    )
    out.append("")
    for flag in flags:
        out.append(f"- {flag['kind']}: " + " / ".join(labels[i] for i in flag["ids"]))
    out.append("")

    excluded = {
        (document_id, m["text"], m["location"])
        for document_id, report in reports.items()
        for m in report["excluded_mentions"]
    }

    def pages(olds_surfaces: set[str]) -> dict[str, list[str]]:
        found: dict[str, set[str]] = defaultdict(set)
        for document_id, extraction in extractions.items():
            for mention in extraction.mentions if extraction else []:
                if mention.text in olds_surfaces:
                    where = mention.location.split(",")[0]
                    note = (
                        ", header/footer, excluded"
                        if (document_id, mention.text, mention.location) in excluded
                        else ""
                    )
                    found[mention.text].add(
                        f"{documents[document_id]} {where} (as {mention.entity_type}{note})"
                    )
        return {k: sorted(v) for k, v in sorted(found.items())}

    out += ["## Merged entities", ""]
    reasons = {m["canonical"]: m for m in analysis["merges"]}
    changed = {
        new: moved[new]
        for new in moved
        if len(moved[new]) > 1 or moved[new][0]["id"] != new
    }
    if not changed:
        out += ["None.", ""]
    for new_id, olds in sorted(changed.items()):
        node = entities[new_id]
        merge = reasons.get(node["name"], {})
        surfaces = {node["name"], *node.get("aliases", [])}
        gone = sorted({o["type"] for o in olds} - set(node.get("observed_types", [])))
        reason = merge.get("method", "")
        if gone:
            reason = "; ".join(
                filter(
                    None,
                    [
                        reason,
                        f"the {', '.join(gone)} node rested only on a running header or "
                        "footer, which is now excluded (the raw result still records that type)",
                    ],
                )
            )
        out += [
            f"### {_name(node)}",
            "",
            "- Previous nodes: " + "; ".join(_name(o) for o in olds),
            f"- Reason: {reason or 'type or spelling now resolved differently'}",
            f"- Observed types kept: {', '.join(node.get('observed_types', []))}",
            "- Sources of every spelling (raw extraction results):",
        ]
        for surface, where in pages(surfaces).items():
            out.append(f"  - '{surface}': {'; '.join(where)}")
        out.append("")
    older = [
        m for m in analysis["merges"] if not any(entities[n]["name"] == m["canonical"] for n in changed)
    ]
    if older:
        out += ["Merges that were already in place before this rebuild:", ""]
        out += [f"- {m['canonical']}: {', '.join(m['merged'])} ({m['method']})" for m in older]
        out.append("")

    out += ["## Removed entities and excluded mentions", ""]
    furniture = {m["text"] for r in reports.values() for m in r["excluded_mentions"]}
    for old in removed:
        why = (
            "every mention is in a running header or footer"
            if old["name"] in furniture
            else "no mention of it remains (check manually)"
        )
        out.append(f"- Removed node {_name(old)}: {why}.")
    if not removed:
        out.append("No entity node was removed.")
    if added:
        out.append("- New nodes: " + "; ".join(_name(e) for e in added))
    out.append("")
    for document_id, report in reports.items():
        for m in report["excluded_mentions"]:
            out.append(
                f"- Mention '{m['text']}' [{m['entity_type']}] in {documents[document_id]}, "
                f"{m['location']}: {m['reason']}."
            )
    out.append("")

    out += ["## Verification", ""]
    out += [f"- {line}" for line in context["checks"]]
    out += [
        "",
        "| Original file (relative to data/) | SHA-256 at backup | Unchanged after rebuild |",
        "|---|---|---|",
    ]
    out += [
        f"| {name} | `{digest[:16]}…` | {'yes' if same else '**NO**'} |"
        for name, digest, same in context["originals"]
    ]
    unchecked = [documents[i] for i, e in extractions.items() if e and not e.facts]
    out += [
        "",
        "## Not covered",
        "",
        f"- No relationship statements exist for {len(unchecked)} document(s), so their files "
        "were not read again and headers or footers were not filtered from their entities: "
        + ("; ".join(unchecked) or "none")
        + ".",
        "- 'Supported' means named together. Negation and the exact meaning of the verb need "
        "your review.",
        "- Names are matched by spelling. 'Agents' or 'agentic systems' are not treated as "
        "'Agentic AI', so some plausible statements are rejected.",
        "- The Graph Explorer draws inferred and supported relationships alike; the status is "
        "stored on every relationship (`support_status`) and shown in retrieved evidence.",
        "- The Documents page still shows the originally parsed text, which was left untouched. "
        "The cleaned text is in `data/exports/layout/`, the verdicts in "
        "`data/exports/validation/`. Locations above refer to the cleaned text.",
        "- Graph analytics (centrality, paths) count inferred relationships as connections.",
        "- The Facts table on Provenance & Conflicts and the Documents page list the raw "
        "extraction, including rejected statements, without a status column.",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", required=True, type=Path)
    arguments = parser.parse_args()

    # No key and no working transport: a model call would fail loudly.
    settings = load_settings(openai_api_key=SecretStr(""))
    MeteredOpenAIProvider._complete = _refuse_model_calls
    configure_logging("ERROR")
    mark = usage_meter.mark()

    recorded = checksums(arguments.backup)
    start = verify_originals(settings.data_dir, recorded)
    if not all(same for _, _, same in start):
        raise SystemExit("Original files differ from the backup; make a new backup first.")
    before = json.loads(
        (arguments.backup / "data" / "exports" / "knowledge_graph.json").read_text(encoding="utf-8")
    )

    lab = KnowledgeLab(settings)
    try:
        decision_query = "MATCH (d:Decision) RETURN count(d) AS n"
        lab.repository.connect()
        decisions = lab.repository.store.execute_query(decision_query)["records"][0]["n"]
        with_facts = [
            d.id
            for d in lab.registry.list_documents()
            if (e := lab.extraction(d.id)) and e.facts
        ]
        other_chunks = [c for c in lab.vector.chunks() if c["document_id"] not in with_facts]

        assembled = lab.rebuild_graph()
        indexed = {i: lab.refresh_search_index(i) for i in with_facts}

        # A second rebuild must change nothing.
        provenance_entries = lab.provenance.verify_integrity()["total_entries"]
        lab.rebuild_graph()
        second_sync = lab.analysis()["last_sync"]
        chain = lab.provenance.verify_integrity()
        stored = lab.graph()

        originals = verify_originals(settings.data_dir, recorded)
        unchanged_chunks = other_chunks == [
            c for c in lab.vector.chunks() if c["document_id"] not in with_facts
        ]
        results = [
            (all(same for _, _, same in originals), f"{len(originals)} original files unchanged"),
            (
                not any(second_sync.values())
                and chain["total_entries"] == provenance_entries,
                "A second rebuild changed nothing in Neo4j or provenance",
            ),
            (chain["valid"], f"Provenance hash chain valid ({chain['total_entries']} entries)"),
            (
                len(stored["entities"]) == len(assembled.entities)
                and len(stored["relationships"]) == len(assembled.relationships),
                "Neo4j holds exactly the rebuilt graph",
            ),
            (
                lab.repository.store.execute_query(decision_query)["records"][0]["n"] == decisions,
                f"Recorded decisions kept ({decisions})",
            ),
            (
                unchanged_chunks,
                f"Search chunks of the other documents unchanged ({len(other_chunks)}); "
                f"re-indexed from cleaned text: {indexed}",
            ),
            (not assembled.unresolved_facts, "Every accepted statement is in the graph"),
        ]
        requests = int(usage_meter.since(mark)["requests"])
        results.append((requests == 0, "No model request was made"))
        report = build_report(
            lab,
            assembled,
            before,
            {
                "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "model_requests": requests,
                "checks": [f"{'PASS' if ok else 'FAIL'}: {text}" for ok, text in results],
                "originals": originals,
            },
        )
    finally:
        lab.close()

    target = settings.exports_dir / "quality" / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8", newline="\n")
    for ok, text in results:
        print(f"{'PASS' if ok else 'FAIL'}: {text}")
    print(f"Report: {target}")
    return 0 if all(ok for ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
