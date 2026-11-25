"""Neo4j persistence through Semantica's ``GraphStore``.

Semantica owns the storage model: nodes are written with ``create_node`` /
``update_node``, relationships with ``create_relationship``, and traversal uses
``get_neighbors`` / ``shortest_path``.

The application adds what 0.7.0 does not provide:

* **upsert** - ``add_nodes``/``add_edges`` always ``CREATE``, so re-running them
  duplicates the graph; we diff by our stable ``id`` property first;
* **safe reads** - ``execute_query`` turns returned nodes into a list of their
  property *names*, so reads go through ``get_nodes``/``get_relationships``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from semantica.graph_store import GraphStore

from backend.config import Settings

logger = logging.getLogger(__name__)

ENTITY_LABEL = "Entity"
MANAGED_BY = "semantica-lab"
_LIMIT = 100_000
_VOLATILE = {"created_at", "updated_at"}


class GraphUnavailableError(RuntimeError):
    """Neo4j cannot be reached; the message never contains credentials."""


@dataclass
class SyncStats:
    nodes_created: int = 0
    nodes_updated: int = 0
    nodes_deleted: int = 0
    relationships_created: int = 0
    relationships_replaced: int = 0
    relationships_deleted: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class GraphRepository:
    def __init__(self, settings: Settings, uri: str | None = None) -> None:
        self.uri = uri or settings.neo4j_uri
        self.store = GraphStore(
            backend="neo4j",
            uri=self.uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password.get_secret_value(),
            database=settings.neo4j_database,
        )
        self._connected = False
        # application id -> node record; refreshed by load() and sync().
        self._node_cache: dict[str, dict] | None = None

    # -------------------------------------------------------------- lifecycle
    def connect(self) -> "GraphRepository":
        if self._connected:
            return self
        try:
            self.store.connect()
            self.store.create_index(ENTITY_LABEL, "id")
        except Exception as exc:
            raise GraphUnavailableError(
                f"Cannot connect to Neo4j at {self.uri}. Is `docker compose up -d` running "
                f"and does NEO4J_PASSWORD match? ({type(exc).__name__})"
            ) from exc
        self._connected = True
        return self

    def close(self) -> None:
        if self._connected:
            self.store.close()
            self._connected = False

    # ------------------------------------------------------------------ reads
    def _nodes(self) -> dict[str, dict]:
        """application id -> {"internal": int, "properties": dict}"""
        rows = self.store.get_nodes(labels=[ENTITY_LABEL], limit=_LIMIT)
        self._node_cache = {
            row["properties"]["id"]: {"internal": row["id"], "properties": row["properties"]}
            for row in rows
            if row["properties"].get("managed_by") == MANAGED_BY
        }
        return dict(self._node_cache)

    def _internal_id(self, entity_id: str) -> int | None:
        if self._node_cache is None or entity_id not in self._node_cache:
            self._nodes()
        node = (self._node_cache or {}).get(entity_id)
        return None if node is None else node["internal"]

    def _relationships(self) -> dict[str, dict]:
        rows = self.store.get_relationships(limit=_LIMIT)
        return {
            row["properties"]["id"]: row
            for row in rows
            if row["properties"].get("managed_by") == MANAGED_BY
        }

    def load(self) -> dict[str, list[dict]]:
        """Read the graph back in the shape Semantica's analysis modules consume."""
        self.connect()
        entities = [
            {k: v for k, v in node["properties"].items() if k != "managed_by"}
            for node in self._nodes().values()
        ]
        relationships = []
        for row in self._relationships().values():
            properties = {k: v for k, v in row["properties"].items() if k != "managed_by"}
            source = properties.pop("source_entity_id")
            target = properties.pop("target_entity_id")
            relationships.append(
                {**properties, "type": row["type"], "source": source, "target": target}
            )
        entities.sort(key=lambda e: e["id"])
        relationships.sort(key=lambda r: (r["source"], r["type"], r["target"]))
        return {"entities": entities, "relationships": relationships}

    def stats(self) -> dict[str, Any]:
        graph = self.load()
        by_type: dict[str, int] = {}
        for entity in graph["entities"]:
            by_type[entity["type"]] = by_type.get(entity["type"], 0) + 1
        by_relation: dict[str, int] = {}
        for relationship in graph["relationships"]:
            by_relation[relationship["type"]] = by_relation.get(relationship["type"], 0) + 1
        return {
            "entities": len(graph["entities"]),
            "relationships": len(graph["relationships"]),
            "entity_types": by_type,
            "relationship_types": by_relation,
        }

    # ------------------------------------------------------------------ write
    def sync(self, graph: dict[str, list[dict]]) -> SyncStats:
        """Make Neo4j match ``graph``: create, update and delete only what changed."""
        self.connect()
        stats = SyncStats()
        now = datetime.now(timezone.utc).isoformat()
        existing_nodes = self._nodes()
        internal_ids: dict[str, int] = {}

        for entity in graph["entities"]:
            desired = _properties(entity)
            current = existing_nodes.pop(entity["id"], None)
            if current is None:
                created = self.store.create_node(
                    [ENTITY_LABEL, entity["type"]],
                    {**desired, "created_at": now, "updated_at": now},
                )
                internal_ids[entity["id"]] = created["id"]
                stats.nodes_created += 1
                continue
            internal_ids[entity["id"]] = current["internal"]
            if _stable(current["properties"]) != _stable(desired):
                created_at = current["properties"].get("created_at", now)
                self.store.update_node(
                    current["internal"],
                    {**desired, "created_at": created_at, "updated_at": now},
                    merge=False,
                )
                stats.nodes_updated += 1

        existing_edges = self._relationships()
        for relationship in graph["relationships"]:
            desired = _properties(
                {
                    **{k: v for k, v in relationship.items() if k not in ("source", "target", "type")},
                    "source_entity_id": relationship["source"],
                    "target_entity_id": relationship["target"],
                }
            )
            current = existing_edges.pop(relationship["id"], None)
            created_at = now
            if current is not None:
                # extracted_at changes on every re-extraction; the claim itself did not.
                if _stable(current["properties"], extra={"extracted_at"}) == _stable(
                    desired, extra={"extracted_at"}
                ):
                    continue
                created_at = current["properties"].get("created_at", now)
                self.store.delete_relationship(current["id"])
                stats.relationships_replaced += 1
            else:
                stats.relationships_created += 1
            self.store.create_relationship(
                internal_ids[relationship["source"]],
                internal_ids[relationship["target"]],
                relationship["type"],
                {**desired, "created_at": created_at, "updated_at": now},
            )

        for stale in existing_edges.values():
            self.store.delete_relationship(stale["id"])
            stats.relationships_deleted += 1
        for stale_node in existing_nodes.values():
            self.store.delete_node(stale_node["internal"], detach=True)
            stats.nodes_deleted += 1
        self._node_cache = None
        logger.info("Neo4j sync: %s", stats.as_dict())
        return stats

    def clear(self) -> None:
        """Delete everything in the configured database (used by reset and tests)."""
        self.connect()
        self.store.execute_query("MATCH (n) DETACH DELETE n")
        self._node_cache = None

    # -------------------------------------------------------------- traversal
    def neighbors(self, entity_id: str, depth: int = 1) -> list[dict]:
        """Entities within ``depth`` hops, via ``GraphStore.get_neighbors``."""
        self.connect()
        internal = self._internal_id(entity_id)
        if internal is None:
            return []
        found = self.store.get_neighbors(internal, depth=depth)
        return [
            {k: v for k, v in row["properties"].items() if k != "managed_by"}
            for row in found
            if row["properties"].get("managed_by") == MANAGED_BY
            and row["properties"]["id"] != entity_id
        ]

    def shortest_path(self, source_id: str, target_id: str, max_depth: int = 6) -> dict | None:
        """Shortest connection between two entities, via ``GraphStore.shortest_path``.

        Decision nodes are not ``managed_by`` this repository; a path that runs
        through one is rejected so evidence only ever comes from extracted facts.
        """
        self.connect()
        start, end = self._internal_id(source_id), self._internal_id(target_id)
        if start is None or end is None or source_id == target_id:
            return None
        path = self.store.shortest_path(start, end, max_depth=max_depth)
        if path is None or any(
            n["properties"].get("managed_by") != MANAGED_BY for n in path["nodes"]
        ):
            return None
        return {
            "length": path["length"],
            "node_ids": [n["properties"]["id"] for n in path["nodes"]],
            "relationship_ids": [r["properties"]["id"] for r in path["relationships"]],
        }


def _properties(record: dict) -> dict:
    """Keep only values Neo4j can store as properties; drop empty ones."""
    properties: dict[str, Any] = {"managed_by": MANAGED_BY}
    for key, value in record.items():
        if value is None or isinstance(value, dict) or value == []:
            continue
        properties[key] = value
    return properties


def _stable(properties: dict, extra: set[str] | None = None) -> dict:
    ignored = _VOLATILE | (extra or set())
    return {k: v for k, v in properties.items() if k not in ignored}
