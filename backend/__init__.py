"""Semantica Knowledge Graph Lab backend.

Environment tweaks that must happen *before* ``semantica`` is imported live here.
"""

from __future__ import annotations

import os
from pathlib import Path

# Semantica's ProgressTracker prints emoji progress bars to the console and each
# component force-enables it; this env var is the supported way to switch it off.
os.environ.setdefault("SEMANTICA_DISABLE_PROGRESS", "1")

# Keep the downloaded embedding model inside the project instead of %TEMP%,
# where Windows cleanup could delete it.
os.environ.setdefault(
    "FASTEMBED_CACHE_PATH", str(Path(__file__).resolve().parents[1] / "data" / "models")
)
