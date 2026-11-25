"""A deliberately small forward-only migration runner.

Each entry in ``MIGRATIONS`` is applied once, in order, and recorded in
``schema_migrations``.  Add a new function for every schema change; never edit
an applied one.  (Alembic would be the next step if the schema starts moving.)
"""

from __future__ import annotations

from typing import Callable

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from database.models import Base


def _001_initial(connection: Connection) -> None:
    Base.metadata.create_all(connection)


MIGRATIONS: list[tuple[str, Callable[[Connection], None]]] = [
    ("001_initial", _001_initial),
]


def apply_migrations(engine: Engine) -> list[str]:
    """Apply pending migrations and return their names."""
    applied_now: list[str] = []
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "name TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
        )
        done = {row[0] for row in connection.execute(text("SELECT name FROM schema_migrations"))}
        for name, migrate in MIGRATIONS:
            if name in done:
                continue
            migrate(connection)
            connection.execute(
                text("INSERT INTO schema_migrations (name) VALUES (:name)"), {"name": name}
            )
            applied_now.append(name)
    return applied_now
