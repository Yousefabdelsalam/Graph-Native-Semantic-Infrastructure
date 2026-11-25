"""Offset-preserving sentence segmentation.

Application-owned on purpose: provenance needs the exact character span of the
statement, and extraction is run per sentence so a relation can never be stitched
together across two unrelated sentences.
"""

from __future__ import annotations

import re

_ABBREVIATIONS = ("Dr.", "Mr.", "Mrs.", "Ms.", "Inc.", "Corp.", "Ltd.", "e.g.", "i.e.", "vs.")
_BOUNDARY = re.compile(r"(?<=[.!?])[ \t]+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, sentence)`` tuples; line breaks always end a sentence."""
    sentences: list[tuple[int, int, str]] = []
    for line in re.finditer(r"[^\n]+", text):
        content, base = line.group(0), line.start()
        cursor = 0
        for boundary in _BOUNDARY.finditer(content):
            head = content[cursor : boundary.start()]
            if head.endswith(_ABBREVIATIONS) and not _ends_sentence(head, content[boundary.end() :]):
                continue
            _append(sentences, text, base + cursor, base + boundary.start())
            cursor = boundary.end()
        _append(sentences, text, base + cursor, base + len(content))
    return sentences


def _ends_sentence(head: str, tail: str) -> bool:
    """'... Northwind Robotics Inc. Northwind Robotics Inc. uses ...' does end."""
    return head.endswith(("Inc.", "Corp.", "Ltd.")) and bool(re.match(r"[A-Z]", tail))


def _append(out: list[tuple[int, int, str]], text: str, start: int, end: int) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end > start:
        out.append((start, end, text[start:end]))
