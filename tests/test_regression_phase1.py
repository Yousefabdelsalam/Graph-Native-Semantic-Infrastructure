"""Phase 2 must not change what Phase 1 did on the original corpus."""

from __future__ import annotations

import json

import pytest

from tests.regression import GOLDEN, build_snapshot, corpus_checksums, pattern_fingerprint


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_original_corpus_and_labels_are_byte_identical(golden, settings) -> None:
    current = corpus_checksums(settings)
    # New files may be added beside the originals; originals may not change.
    assert {name: current.get(name) for name in golden["checksums"]} == golden["checksums"]


def test_regex_pattern_pack_is_frozen(golden) -> None:
    assert pattern_fingerprint() == golden["pattern_fingerprint"], (
        "The regex pattern pack changed. The benchmark's regex results are only meaningful "
        "if the pack predates the benchmark."
    )


@pytest.fixture(scope="module")
def snapshot(loaded_lab) -> dict:
    return build_snapshot(loaded_lab)


@pytest.mark.parametrize(
    "section",
    [
        "facts",
        "entities",
        "relationships",
        "conflicts",
        "successions",
        "merges",
        "extraction_scores",
        "retrieval",
    ],
)
def test_behaviour_on_original_corpus_is_unchanged(golden, snapshot, section: str) -> None:
    assert snapshot[section] == golden[section]
