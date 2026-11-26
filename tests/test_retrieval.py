"""Vector, graph, hybrid and native retrieval over the sample corpus."""

from __future__ import annotations

import pytest

from backend.evaluation.metrics import extraction_scores, score_result
from backend.evaluation.runner import evaluate_extraction, evaluate_retrieval, load_questions
from backend.retrieval.answer import (
    INSUFFICIENT,
    AnswerUnavailableError,
    build_prompt,
    generate_answer,
)
from backend.retrieval.engine import MODES
from backend.schemas import Evidence, RetrievalResult


@pytest.fixture(scope="module")
def engine(loaded_lab):
    return loaded_lab.engine()


def test_vector_search_finds_the_supporting_chunk(engine) -> None:
    result = engine.retrieve("Which technologies does VisionCore depend on?", "vector", top_k=3)
    assert "03_technology_stack.docx" in result.source_files
    top = engine.retrieve("Northwind Robotics warehouse automation", "vector", top_k=1)
    assert top.source_files == ["05_integrations.txt"]
    scores = [e.score for e in result.evidence]
    assert scores == sorted(scores, reverse=True) and all(-1 <= s <= 1 for s in scores)
    assert all(e.location and e.document_ids for e in result.evidence)


def test_graph_retrieval_returns_sourced_facts(engine) -> None:
    result = engine.retrieve("Which projects use PostgreSQL?", "graph")
    text = " ".join(e.text for e in result.evidence)
    assert "Project Atlas USES PostgreSQL" in text and "Project Cobalt DEPENDS_ON" in text
    assert all(e.filenames and e.location for e in result.evidence)
    assert result.sufficient and result.entities and result.relationships


def test_multi_hop_path_between_named_entities(engine) -> None:
    result = engine.retrieve("How is Northwind Robotics connected to PostgreSQL?", "graph")
    assert result.paths and result.paths[0].startswith("Northwind Robotics")
    assert result.paths[0].endswith("PostgreSQL") and result.paths[0].count("[") >= 3


def test_hop_limit_restricts_traversal(engine) -> None:
    near = engine.retrieve("What about MongoDB and Kafka?", "graph", max_hops=1)
    far = engine.retrieve("What about MongoDB and Kafka?", "graph", max_hops=3, max_facts=40)
    assert len(far.relationships) > len(near.relationships)


def test_entity_linking_uses_aliases_and_longest_match(engine) -> None:
    linked = engine.link_entities("Does Sarah lead the VisionCore Team at Acme AI Inc.?")
    assert linked == ["person:sarah-chen", "team:visioncore-team", "organization:acme-ai"]


def test_temporal_filter_removes_later_facts(engine) -> None:
    result = engine.retrieve("Who led the Platform Team in 2024?", "graph")
    text = " ".join(e.text for e in result.evidence)
    assert "Marcus Webb LEADS Platform Team" in text
    assert "Priya Raman LEADS" not in text


def test_conflicts_are_surfaced_with_both_sources(engine) -> None:
    result = engine.retrieve("Which facts about VisionCore have conflicting sources?", "graph")
    conflicts = [e for e in result.evidence if e.kind == "conflict"]
    assert len(conflicts) == 3
    assert all(len(e.filenames) == 2 for e in conflicts)


def test_hybrid_combines_chunks_and_facts(engine) -> None:
    result = engine.retrieve("Who works with Sarah?", "hybrid", top_k=2, max_facts=5)
    kinds = [e.kind for e in result.evidence]
    assert kinds.count("chunk") == 2 and kinds.count("fact") >= 5


def test_native_context_retriever_runs_and_cites_documents(engine) -> None:
    result = engine.retrieve("What does DocuMind depend on?", "semantica")
    assert {e.kind for e in result.evidence} == {"chunk", "context"}
    assert result.source_files


@pytest.mark.parametrize("mode", ["vector", "graph", "hybrid"])
def test_unanswerable_question_is_declared_insufficient(engine, settings, mode: str) -> None:
    result = engine.retrieve("What is the annual revenue of Globex Corporation?", mode)
    assert not result.sufficient and result.note
    assert generate_answer(settings, result) == INSUFFICIENT  # no model call is made


def test_invalid_requests_are_rejected(engine) -> None:
    with pytest.raises(ValueError):
        engine.retrieve("   ", "graph")
    with pytest.raises(ValueError):
        engine.retrieve("q", "magic")


def _result() -> RetrievalResult:
    return RetrievalResult(
        question="Who leads?",
        mode="graph",
        latency_ms=1,
        evidence=[Evidence(kind="fact", text="A LEADS B", filenames=["a.txt"], reference="r")],
    )


def test_answer_generation_needs_a_key(settings) -> None:
    with pytest.raises(AnswerUnavailableError):
        generate_answer(settings, _result())


def test_answer_generation_sends_only_the_evidence(monkeypatch, tmp_path) -> None:
    from pydantic import SecretStr

    from backend.config import load_settings

    sent = {}

    class FakeProvider:
        def __init__(self, **kwargs) -> None:
            sent["model"] = kwargs["model"]

        def generate(self, prompt: str, **kwargs) -> str:
            sent["prompt"] = prompt
            return " A leads B [1]. "

    monkeypatch.setattr("backend.retrieval.answer.MeteredOpenAIProvider", FakeProvider)
    configured = load_settings(data_dir=tmp_path, openai_api_key=SecretStr("test-key"))
    assert generate_answer(configured, _result()) == "A leads B [1]."
    assert sent["prompt"] == build_prompt(_result())
    assert "[1] (a.txt) A LEADS B" in sent["prompt"]
    assert "test-key" not in sent["prompt"] and "test-key" not in repr(configured)


def test_extraction_evaluation_matches_labels(settings) -> None:
    report = evaluate_extraction(settings, ["pattern", "regex"])
    assert report["pattern"]["correct"] == 0
    assert report["regex"]["precision"] == 1.0 and report["regex"]["recall"] >= 0.95
    assert [g.get("hard") for g in report["regex"]["missed"]] == [True]


def test_retrieval_evaluation_covers_every_question_and_mode(loaded_lab, engine) -> None:
    questions = load_questions(loaded_lab.settings)
    assert len(questions) >= 10
    report = evaluate_retrieval(engine, loaded_lab.settings, questions)
    assert len(report["rows"]) == len(questions) * len(MODES)
    assert set(report["by_mode"]) == set(MODES)
    assert all(r["answer"] is None for r in report["rows"])


def test_scoring_rules() -> None:
    question = {
        "id": "x",
        "category": "temporal",
        "expected": [["Marcus"]],
        "excluded": ["Priya"],
        "sources": ["a.txt"],
    }
    good = _result().model_copy(
        update={"evidence": [Evidence(kind="fact", text="Marcus led", filenames=["a.txt"], reference="r")]}
    )
    assert score_result(question, good)["correct"]
    leaked = good.model_copy(
        update={"evidence": [*good.evidence, Evidence(kind="fact", text="Priya leads", reference="s")]}
    )
    assert not score_result(question, leaked)["correct"]
    assert not score_result(question, good.model_copy(update={"sufficient": False}))["correct"]


def test_symmetric_relations_score_in_either_direction() -> None:
    gold = [{"document": "d", "subject": "A", "predicate": "COLLABORATES_WITH", "object": "B"}]
    scores = extraction_scores({("d", "B", "COLLABORATES_WITH", "A")}, gold, {"COLLABORATES_WITH"})
    assert scores["precision"] == scores["recall"] == 1.0
