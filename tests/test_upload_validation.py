"""Upload validation (pure application code)."""

from __future__ import annotations

import io
import zipfile

import pytest

from backend.ingestion.validation import (
    UploadValidationError,
    safe_display_name,
    validate_upload,
)

LIMIT = 1_048_576


def test_accepts_text_and_hashes_content() -> None:
    first = validate_upload("notes.txt", b"Acme AI developed VisionCore.", LIMIT)
    second = validate_upload("copy.txt", b"Acme AI developed VisionCore.", LIMIT)
    assert first.sha256 == second.sha256
    assert first.stored_filename == second.stored_filename
    assert first.extension == ".txt"


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("script.exe", b"MZ"),
        ("empty.txt", b""),
        ("fake.pdf", b"this is not a pdf"),
        ("fake.docx", b"this is not a zip"),
        ("binary.txt", b"abc\x00def"),
    ],
)
def test_rejects_malformed_or_unsupported(name: str, content: bytes) -> None:
    with pytest.raises(UploadValidationError):
        validate_upload(name, content, LIMIT)


def test_rejects_oversized_file() -> None:
    with pytest.raises(UploadValidationError, match="limit"):
        validate_upload("big.txt", b"a" * (LIMIT + 1), LIMIT)


def test_rejects_zip_that_is_not_a_word_document() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("hello.txt", "hi")
    with pytest.raises(UploadValidationError):
        validate_upload("zipped.docx", buffer.getvalue(), LIMIT)


@pytest.mark.parametrize(
    "name", ["../../etc/passwd.txt", "..\\..\\windows\\passwd.txt", "C:\\tmp\\passwd.txt"]
)
def test_path_components_are_stripped(name: str) -> None:
    assert safe_display_name(name) == "passwd.txt"
    assert "/" not in validate_upload(name, b"x", LIMIT).stored_filename
