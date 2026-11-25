"""Upload validation (application-owned; runs before Semantica sees a file)."""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath

from backend.config import SUPPORTED_EXTENSIONS


class UploadValidationError(ValueError):
    """The uploaded file is not acceptable; the message is safe to show users."""


@dataclass(frozen=True)
class ValidatedUpload:
    original_filename: str
    extension: str
    sha256: str
    size: int

    @property
    def stored_filename(self) -> str:
        """Content-addressed name: never derived from user-controlled text."""
        return f"{self.sha256[:20]}{self.extension}"


def safe_display_name(filename: str) -> str:
    """Strip any path components and control characters from a client filename."""
    name = PurePath(filename.replace("\\", "/")).name
    name = re.sub(r"[\x00-\x1f]", "", name).strip()
    return name[:255] or "unnamed"


def validate_upload(filename: str, content: bytes, max_bytes: int) -> ValidatedUpload:
    """Check extension, size and that the bytes look like the claimed format."""
    name = safe_display_name(filename)
    extension = PurePath(name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError(
            f"Unsupported file type '{extension or '(none)'}'. "
            f"Allowed: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    if not content:
        raise UploadValidationError(f"'{name}' is empty.")
    if len(content) > max_bytes:
        raise UploadValidationError(
            f"'{name}' is {len(content) / 1_048_576:.1f} MB; "
            f"the limit is {max_bytes / 1_048_576:.0f} MB."
        )

    if extension == ".pdf" and not content.startswith(b"%PDF-"):
        raise UploadValidationError(f"'{name}' is not a valid PDF file.")
    if extension == ".docx":
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                if "word/document.xml" not in archive.namelist():
                    raise UploadValidationError(f"'{name}' is not a valid DOCX file.")
                # Guard against decompression bombs.
                if sum(info.file_size for info in archive.infolist()) > max_bytes * 20:
                    raise UploadValidationError(f"'{name}' expands to an unreasonable size.")
        except zipfile.BadZipFile as exc:
            raise UploadValidationError(f"'{name}' is not a valid DOCX file.") from exc
    if extension in (".txt", ".html", ".htm") and b"\x00" in content:
        raise UploadValidationError(f"'{name}' looks like a binary file, not text.")

    return ValidatedUpload(name, extension, hashlib.sha256(content).hexdigest(), len(content))
