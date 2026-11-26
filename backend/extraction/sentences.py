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


def unwrap_lines(text: str, long_line: int = 50) -> str:
    """Join lines broken by page layout, without moving any character offsets.

    PDF text wraps mid-sentence.  A line break becomes a space when the line does
    not end a sentence and either looks full or is continued in lower case; short
    lines such as headings stay separate.  The result has the same length.
    """
    characters = list(text)
    start = 0
    while (end := text.find("\n", start)) != -1:
        line = text[start:end].rstrip()
        following = text[end + 1 : end + 2]
        if (
            line
            and following not in ("", "\n")
            and not line.endswith((".", "!", "?", ":"))
            and (len(line) >= long_line or following.islower())
        ):
            characters[end] = " "
        start = end + 1
    return "".join(characters)
