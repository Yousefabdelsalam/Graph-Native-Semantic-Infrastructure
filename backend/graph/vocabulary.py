"""Curated name knowledge used by entity resolution (application configuration).

The alias table is handed to Semantica's ``EntityNormalizer``.  It encodes facts
no string-similarity measure can know reliably, for example that "Postgres" and
"PostgreSQL" are one product.  Anything *not* listed here is left to Semantica's
``DuplicateDetector`` and surfaced for review.
"""

from __future__ import annotations

# lower-cased alias -> canonical name
ALIASES: dict[str, str] = {
    "postgres": "PostgreSQL",
    "acme artificial intelligence": "Acme AI",
    # The agent pattern "ReAct" (Reason + Act) and the names documents give it.
    "react framework": "ReAct",
    "react loop": "ReAct",
}

LEGAL_SUFFIXES: tuple[str, ...] = ("inc.", "inc", "corp.", "corp", "ltd.", "ltd", "llc")
HONORIFICS: tuple[str, ...] = ("dr.", "prof.", "mr.", "mrs.", "ms.")
