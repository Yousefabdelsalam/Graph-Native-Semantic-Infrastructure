"""Test double for the Neo4j repository, so unit tests need no database.

It mirrors the public behaviour of ``GraphRepository``; the real class is
covered by the integration tests.
"""

from __future__ import annotations

import copy
from collections import deque

from backend.graph.repository import GraphUnavailableError, SyncStats


class InMemoryRepository:
    def __init__(self) -> None:
        self.graph: dict[str, list[dict]] = {"entities": [], "relationships": []}
        self.fail = False

    def connect(self) -> "InMemoryRepository":
        if self.fail:
            raise GraphUnavailableError("Cannot connect to Neo4j (simulated).")
        return self

    def close(self) -> None:
        pass

    def sync(self, graph: dict[str, list[dict]]) -> SyncStats:
        self.connect()
        old_nodes = {e["id"] for e in self.graph["entities"]}
        new_nodes = {e["id"] for e in graph["entities"]}
        old_edges = {r["id"] for r in self.graph["relationships"]}
        new_edges = {r["id"] for r in graph["relationships"]}
        self.graph = {
            key: [
                {k: v for k, v in row.items() if v is not None and v != []}
                for row in copy.deepcopy(graph[key])
            ]
            for key in ("entities", "relationships")
        }
        return SyncStats(
            nodes_created=len(new_nodes - old_nodes),
            nodes_deleted=len(old_nodes - new_nodes),
            relationships_created=len(new_edges - old_edges),
            relationships_deleted=len(old_edges - new_edges),
        )

    def load(self) -> dict[str, list[dict]]:
        self.connect()
        return copy.deepcopy(self.graph)

    def clear(self) -> None:
        self.graph = {"entities": [], "relationships": []}

    def _walk(self, start: str, depth: int) -> dict[str, tuple[list[str], list[str]]]:
        found = {start: ([start], [])}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            nodes, edges = found[current]
            if len(edges) >= depth:
                continue
            for rel in self.graph["relationships"]:
                if current not in (rel["source"], rel["target"]):
                    continue
                other = rel["target"] if rel["source"] == current else rel["source"]
                if other not in found:
                    found[other] = (nodes + [other], edges + [rel["id"]])
                    queue.append(other)
        return found

    def neighbors(self, entity_id: str, depth: int = 1) -> list[dict]:
        reached = self._walk(entity_id, depth)
        return [e for e in self.graph["entities"] if e["id"] in reached and e["id"] != entity_id]

    def shortest_path(self, source_id: str, target_id: str, max_depth: int = 6) -> dict | None:
        found = self._walk(source_id, max_depth).get(target_id)
        if source_id == target_id or found is None:
            return None
        return {"length": len(found[1]), "node_ids": found[0], "relationship_ids": found[1]}
