"""AI extraction with a stand-in OpenAI client: no key, no network, no cost."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx2
import openai
import pytest
from pydantic import SecretStr
from semantica.semantic_extract import NERExtractor, providers

from backend.extraction.extractor import (
    ExtractionError,
    ExtractionPipeline,
    locate_evidence,
    passages,
)
from backend.extraction.sentences import unwrap_lines
from backend.extraction.validation import validate
from backend.graph.builder import assemble_graph
from backend.llm.provider import MAX_ATTEMPTS, MeteredOpenAIProvider, register, usage_meter
from backend.ontology.service import OntologyService
from backend.services.lab import KnowledgeLab
from tests.conftest import make_document, make_settings
from tests.fakes import InMemoryRepository

TEXT = "Acme AI developed VisionCore."
ENTITIES = {
    "entities": [
        {"text": "Acme AI", "label": "Organization", "confidence": 0.9},
        {"text": "VisionCore", "label": "Product", "confidence": 0.9},
    ]
}
RELATIONS = {
    "relations": [
        {"subject": "Acme AI", "object": "VisionCore", "predicate": "DEVELOPED", "confidence": 0.9}
    ]
}


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(kind: type[openai.APIStatusError], status: int, headers=None, body=None):
    response = httpx2.Response(status, request=_request(), headers=headers or {})
    return kind("simulated", response=response, body=body)


class FakeOpenAI:
    """Answers entity and relation prompts, or raises the configured error."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.error: Exception | None = None
        self.replies = {"entities": ENTITIES, "relations": RELATIONS}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        prompt = request["messages"][0]["content"]
        kind = "relations" if "Entities found in text" in prompt else "entities"
        reply = self.replies[kind]
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )


@pytest.fixture()
def fake(monkeypatch) -> FakeOpenAI:
    client = FakeOpenAI()
    waits: list[float] = []
    monkeypatch.setattr(
        MeteredOpenAIProvider, "_init_client", lambda self: setattr(self, "client", client)
    )
    monkeypatch.setattr(MeteredOpenAIProvider, "sleep", staticmethod(waits.append))
    client.waits = waits
    providers._provider_pool.clear()
    yield client
    providers._provider_pool.clear()


@pytest.fixture()
def llm_settings(tmp_path):
    return make_settings(tmp_path).model_copy(update={"openai_api_key": SecretStr("test-key")})


def _extract(settings, text: str):
    return ExtractionPipeline(settings, "llm").extract(make_document(text))


def test_extracts_facts_and_meters_tokens(fake, llm_settings) -> None:
    mark = usage_meter.mark()
    extraction = _extract(llm_settings, TEXT)
    assert [(f.subject, f.predicate, f.object) for f in extraction.facts] == [
        ("Acme AI", "DEVELOPED", "VisionCore")
    ]
    assert extraction.facts[0].method == "llm" and extraction.facts[0].quote == TEXT
    used = usage_meter.since(mark)
    assert (used["requests"], used["prompt_tokens"], used["completion_tokens"]) == (2, 200, 40)
    assert all(r["model"] == llm_settings.openai_model for r in fake.requests)
    assert all(r["reasoning_effort"] == "low" for r in fake.requests)
    assert all("temperature" not in r for r in fake.requests)


def test_types_are_restricted_by_a_strict_schema(fake, llm_settings) -> None:
    _extract(llm_settings, TEXT)
    schemas = [r["response_format"]["json_schema"] for r in fake.requests]
    assert all(s["strict"] for s in schemas)
    entity_items = schemas[0]["schema"]["properties"]["entities"]["items"]
    relation_items = schemas[1]["schema"]["properties"]["relations"]["items"]
    assert "Concept" in entity_items["properties"]["label"]["enum"]
    assert {"USES", "ENABLES", "RELATED_TO"} <= set(
        relation_items["properties"]["predicate"]["enum"]
    )


def test_concepts_from_a_wrapped_pdf_line_become_graph_relationships(fake, llm_settings) -> None:
    text = (
        "4. PLANNING\n"
        "Agentic systems rely on explicit planning frameworks that decompose broad\n"
        "goals into smaller steps, and this planning enables reliable tool use across\n"
        "long tasks."
    )
    fake.replies = {
        "entities": {
            "entities": [
                {"text": "agentic systems", "label": "Concept", "confidence": 0.9},
                {"text": "planning frameworks", "label": "Concept", "confidence": 0.9},
                {"text": "tool use", "label": "Concept", "confidence": 0.9},
            ]
        },
        "relations": {
            "relations": [
                {
                    "subject": "agentic systems",
                    "predicate": "USES",
                    "object": "planning frameworks",
                    "confidence": 0.9,
                },
                {
                    "subject": "planning frameworks",
                    "predicate": "ENABLES",
                    "object": "tool use",
                    "confidence": 0.8,
                },
            ]
        },
    }
    extraction = _extract(llm_settings, text)
    quotes = {f.predicate: f.quote for f in extraction.facts}
    assert set(quotes) == {"USES", "ENABLES"}
    # The sentence is recovered whole although the PDF broke it over three lines.
    assert quotes["USES"].startswith("Agentic systems rely on explicit planning frameworks")
    assert quotes["USES"].endswith("long tasks.") and "PLANNING" not in quotes["USES"]
    assert extraction.facts[0].subject == "Agentic systems"  # the document's spelling

    # A model result must be validated against the document before it is graphed.
    with pytest.raises(ValueError, match="not been validated"):
        assemble_graph([extraction])
    report = validate(extraction, make_document(text))
    assert {v.status for v in report.verdicts} == {"supported"}
    assembled = assemble_graph([report.extraction])
    assert {r["type"] for r in assembled.relationships} == {"USES", "ENABLES"}
    service = OntologyService()
    assert service.validate(assembled.as_dict(), service.corrected_shacl())["conforms"]


def test_names_absent_from_the_text_are_rejected(fake, llm_settings) -> None:
    fake.replies["entities"] = {
        "entities": [
            *ENTITIES["entities"],
            {"text": "Invented Corp", "label": "Organization", "confidence": 0.99},
        ]
    }
    extraction = _extract(llm_settings, TEXT)
    assert "Invented Corp" not in {m.text for m in extraction.mentions}
    assert extraction.discarded["entity not found in text"] == 1


def test_person_only_relations_are_enforced(fake, llm_settings) -> None:
    fake.replies["relations"] = {
        "relations": [
            {"subject": "VisionCore", "object": "Acme AI", "predicate": "LEADS", "confidence": 0.9}
        ]
    }
    extraction = _extract(llm_settings, TEXT)
    assert extraction.facts == []
    assert extraction.discarded["domain/range violation: LEADS"] == 1


def test_text_without_relations_costs_no_extra_request(fake, llm_settings) -> None:
    fake.replies["relations"] = {"relations": []}
    extraction = _extract(llm_settings, TEXT)
    assert extraction.facts == [] and len(fake.requests) == 2


@pytest.mark.parametrize(
    ("error", "attempts"),
    [
        (_status_error(openai.RateLimitError, 429), MAX_ATTEMPTS),
        (openai.APIConnectionError(request=_request()), MAX_ATTEMPTS),
        (_status_error(openai.RateLimitError, 429, body={"code": "insufficient_quota"}), 1),
        (_status_error(openai.AuthenticationError, 401), 1),
        (_status_error(openai.NotFoundError, 404), 1),
    ],
    ids=["rate limit", "network", "no credit", "bad key", "unknown model"],
)
def test_failures_raise_after_bounded_attempts(fake, llm_settings, error, attempts) -> None:
    fake.error = error
    with pytest.raises(ExtractionError) as raised:
        _extract(llm_settings, TEXT)
    assert len(fake.requests) == attempts
    assert "test-key" not in str(raised.value)


def test_retry_after_header_sets_the_wait(fake, llm_settings) -> None:
    fake.error = _status_error(openai.RateLimitError, 429, {"retry-after": "7"})
    with pytest.raises(ExtractionError):
        _extract(llm_settings, TEXT)
    assert fake.waits == [7.0] * (MAX_ATTEMPTS - 1)


def test_unusable_output_raises(fake, llm_settings) -> None:
    fake.replies["entities"] = "not json at all"
    with pytest.raises(ExtractionError, match="schema"):
        _extract(llm_settings, TEXT)
    assert len(fake.requests) <= MAX_ATTEMPTS


def test_failed_reprocessing_keeps_existing_facts(fake, llm_settings) -> None:
    lab = KnowledgeLab(llm_settings, repository=InMemoryRepository())
    try:
        outcome = lab.ingest("a.txt", TEXT.encode(), method="llm")
        assert outcome.status == "processed"
        before = (lab.graph(), lab.extraction(outcome.document_id))
        fact_id = before[1].facts[0].fact_id

        fake.error = _status_error(openai.RateLimitError, 429)
        failed = lab.process(outcome.document_id, method="llm")

        assert failed.status == "failed"
        assert "LLM extraction failed" in failed.message
        assert (lab.graph(), lab.extraction(outcome.document_id)) == before
        assert not lab.provenance.get(fact_id).get("invalidated")
    finally:
        lab.close()


def test_semantica_extractor_classes_swallow_llm_errors(fake, llm_settings) -> None:
    """Why the app calls the LLM functions directly (Semantica 0.7.0 behaviour)."""
    fake.error = _status_error(openai.AuthenticationError, 401)
    extractor = NERExtractor(
        method="llm", provider=register(), llm_model="m", api_key="test-key"
    )
    results = extractor.extract_entities(TEXT)  # no exception, other methods took over
    assert fake.requests
    assert all(not e.metadata["extraction_method"].startswith("llm") for e in results)


# -- text preparation -------------------------------------------------------------
def test_unwrapping_keeps_offsets_and_headings() -> None:
    text = "INTRODUCTION\nAgents plan their work and then call\nexternal tools.\nNext heading"
    joined = unwrap_lines(text)
    assert len(joined) == len(text)
    assert joined.split("\n") == [
        "INTRODUCTION",
        "Agents plan their work and then call external tools.",
        "Next heading",
    ]


def test_passages_cover_the_document_in_bounded_pieces() -> None:
    document = make_document(" ".join(f"Sentence number {i} is here." for i in range(200)))
    pieces = passages(document, size=300)
    assert len(pieces) > 5 and all(len(text) <= 300 for _, _, text in pieces)
    assert all(document.text[start:end] == text for start, end, text in pieces)
    assert pieces[-1][1] == len(document.text)


def test_evidence_spans_two_sentences_when_needed() -> None:
    passage = "Memory stores past steps. It lets the planner recover from errors."
    quote, start, end = locate_evidence(passage, "Memory", "planner")
    assert quote == passage and (start, end) == (0, len(passage))
