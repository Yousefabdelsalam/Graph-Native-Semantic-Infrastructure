"""Application settings, loaded from environment variables / ``.env``."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt", ".html", ".htm")


class Settings(BaseModel):
    """Validated runtime configuration. Secrets are ``SecretStr`` so they never log."""

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")
    neo4j_database: str = "neo4j"

    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "llama-3.3-70b-versatile"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    data_dir: Path = PROJECT_ROOT / "data"
    max_upload_mb: int = Field(default=10, ge=1, le=100)
    log_level: str = "INFO"

    @field_validator("data_dir", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def sample_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "sample_documents"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "vector_index"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def extractions_dir(self) -> Path:
        return self.exports_dir / "extractions"

    @property
    def metadata_db(self) -> Path:
        return self.data_dir / "app_metadata.db"

    @property
    def provenance_db(self) -> Path:
        return self.data_dir / "provenance.db"

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "models"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def llm_available(self) -> bool:
        return bool(self.groq_api_key.get_secret_value())

    def ensure_directories(self) -> None:
        for directory in (
            self.uploads_dir,
            self.vector_dir,
            self.extractions_dir,
            self.models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


_ENV_MAP = {
    "neo4j_uri": "NEO4J_URI",
    "neo4j_user": "NEO4J_USER",
    "neo4j_password": "NEO4J_PASSWORD",
    "neo4j_database": "NEO4J_DATABASE",
    "groq_api_key": "GROQ_API_KEY",
    "groq_model": "GROQ_MODEL",
    "embedding_model": "EMBEDDING_MODEL",
    "data_dir": "DATA_DIR",
    "max_upload_mb": "MAX_UPLOAD_MB",
    "log_level": "LOG_LEVEL",
}


def load_settings(**overrides: object) -> Settings:
    """Read ``.env`` + environment, then apply explicit overrides (used by tests)."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    values: dict[str, object] = {
        field: os.environ[env] for field, env in _ENV_MAP.items() if os.environ.get(env)
    }
    values.update(overrides)
    return Settings(**values)


def configure_logging(level: str = "INFO") -> None:
    """Console logging for the app; keeps third-party libraries quiet."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    for noisy in ("neo4j", "httpx", "urllib3", "fastembed", "pdfminer"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    # Semantica logs every internal step at INFO.
    logging.getLogger("semantica").setLevel(logging.WARNING)
