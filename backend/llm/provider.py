"""A metered OpenAI provider, plugged into Semantica's ``provider_registry``.

Semantica 0.7.0's ``OpenAIProvider`` defaults to a model OpenAI has shut down,
discards token usage and, on failure, retries a request up to six times on top of
the SDK's own retries.  This subclass is used through Semantica's supported
extension point, so ``extract_entities_llm`` / ``extract_relations_llm`` still
build the prompts and parse the results.  It changes only the transport:

* one HTTP request per attempt, at most ``MAX_ATTEMPTS`` per call;
* back-off for rate limits and server errors, honouring ``retry-after``;
* non-retryable errors (bad key, no credit, unknown model) fail immediately;
* strict structured output, so entity and relation types stay inside the ontology;
* every response's token usage is recorded in :data:`usage_meter`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import openai
from pydantic import BaseModel
from semantica.semantic_extract.methods import configure_cache
from semantica.semantic_extract.providers import OpenAIProvider
from semantica.semantic_extract.registry import provider_registry

logger = logging.getLogger(__name__)

PROVIDER_NAME = "openai-lab"
MAX_ATTEMPTS = 3
MAX_WAIT_SECONDS = 90.0

_RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)


class LLMError(RuntimeError):
    """A model call failed.  The message is safe to log and show."""


@dataclass(frozen=True)
class Usage:
    model: str
    prompt_tokens: int
    completion_tokens: int  # includes reasoning tokens
    seconds: float  # time of the successful request, excluding back-off


class UsageMeter:
    """Thread-safe record of API-reported token usage."""

    def __init__(self) -> None:
        self._records: list[Usage] = []
        self._lock = threading.Lock()

    def add(self, usage: Usage) -> None:
        with self._lock:
            self._records.append(usage)

    def mark(self) -> int:
        with self._lock:
            return len(self._records)

    def since(self, mark: int = 0) -> dict[str, float]:
        with self._lock:
            records = self._records[mark:]
        return {
            "requests": len(records),
            "prompt_tokens": sum(r.prompt_tokens for r in records),
            "completion_tokens": sum(r.completion_tokens for r in records),
            "request_seconds": round(sum(r.seconds for r in records), 3),
        }


usage_meter = UsageMeter()


def _retry_delay(error: Exception, attempt: int) -> float:
    response = getattr(error, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    try:
        delay = float(header) if header else 2.0**attempt
    except ValueError:
        delay = 2.0**attempt
    return min(delay, MAX_WAIT_SECONDS)


def _strict_list(field: str, item: dict[str, dict]) -> dict:
    """A strict JSON-schema response format: ``{field: [item, ...]}``."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"extracted_{field}",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [field],
                "properties": {
                    field: {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": list(item),
                            "properties": item,
                        },
                    }
                },
            },
        },
    }


def response_format(schema: type[BaseModel], options: dict) -> dict:
    """Restrict labels and predicates to the allowed names Semantica was given."""
    text, number = {"type": "string"}, {"type": "number"}
    fields = schema.model_fields
    if "entities" in fields and options.get("entity_types"):
        label = {"type": "string", "enum": list(options["entity_types"])}
        return _strict_list("entities", {"text": text, "label": label, "confidence": number})
    if "relations" in fields and options.get("relation_types"):
        predicate = {"type": "string", "enum": list(options["relation_types"])}
        return _strict_list(
            "relations",
            {"subject": text, "predicate": predicate, "object": text, "confidence": number},
        )
    return {"type": "json_object"}


class MeteredOpenAIProvider(OpenAIProvider):
    sleep: Callable[[float], None] = staticmethod(time.sleep)

    def __init__(self, *args: Any, reasoning_effort: str | None = None, **kwargs: Any) -> None:
        self.reasoning_effort = reasoning_effort
        super().__init__(*args, **kwargs)

    def _init_client(self) -> None:
        # Retries are handled here so that they are bounded and visible.
        self.client = openai.OpenAI(api_key=self.api_key, max_retries=0) if self.api_key else None

    def _complete(self, prompt: str, *, output: dict | None, options: dict) -> str:
        if self.client is None:
            raise LLMError("OpenAI client not initialised: OPENAI_API_KEY is missing.")
        request: dict[str, Any] = {
            "model": options.get("model") or self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        # No temperature unless asked for: reasoning models may restrict it.
        self._add_if_set(request, options, "temperature", "max_completion_tokens", "seed")
        if effort := options.get("reasoning_effort") or self.reasoning_effort:
            request["reasoning_effort"] = effort
        if output:
            request["response_format"] = output

        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(**request)
            except openai.APIError as error:
                # "insufficient_quota" is reported as a rate limit but never recovers.
                no_credit = getattr(error, "code", None) == "insufficient_quota"
                if no_credit:
                    raise LLMError(
                        "OpenAI reports no remaining credit (insufficient_quota). "
                        "Check billing on your OpenAI account."
                    ) from error
                if not isinstance(error, _RETRYABLE):
                    # The server's explanation helps with a bad parameter or model
                    # name.  Authentication messages are left out: they quote the key.
                    detail = ""
                    if isinstance(error, (openai.BadRequestError, openai.NotFoundError)):
                        detail = f" OpenAI said: {error.message[:200]}"
                    raise LLMError(
                        f"OpenAI rejected the request ({type(error).__name__}, model "
                        f"'{request['model']}'). Check the API key and OPENAI_MODEL.{detail}"
                    ) from error
                if attempt == MAX_ATTEMPTS:
                    raise LLMError(
                        f"OpenAI request failed after {attempt} attempts ({type(error).__name__})."
                    ) from error
                delay = _retry_delay(error, attempt)
                logger.warning(
                    "OpenAI %s; waiting %.0fs (attempt %d/%d)",
                    type(error).__name__,
                    delay,
                    attempt,
                    MAX_ATTEMPTS,
                )
                self.sleep(delay)
                continue
            usage = response.usage
            usage_meter.add(
                Usage(
                    model=request["model"],
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    seconds=time.perf_counter() - started,
                )
            )
            return response.choices[0].message.content or ""
        raise AssertionError("unreachable")

    # -- Semantica provider interface -----------------------------------------
    def generate(self, prompt: str, **kwargs) -> str:
        return self._complete(prompt, output=None, options=kwargs)

    def generate_typed(
        self, prompt: str, schema: type[BaseModel], max_retries: int = MAX_ATTEMPTS, **kwargs
    ) -> BaseModel:
        """Structured generation validated against Semantica's schema."""
        current = prompt if "json" in prompt.lower() else f"{prompt}\n\nReturn valid JSON only."
        output = response_format(schema, kwargs)
        attempts = max(1, min(max_retries, MAX_ATTEMPTS))
        problem: Exception | None = None
        for _ in range(attempts):
            content = self._complete(current, output=output, options=kwargs)
            try:
                return schema.model_validate(self._parse_json(content))
            except Exception as error:  # invalid JSON or a schema mismatch
                problem = error
                current = (
                    f"{prompt}\n\nThe previous reply was not valid for the schema: "
                    f"{str(error)[:400]}\nReturn corrected JSON only."
                )
        raise LLMError(
            f"Model output did not match the schema after {attempts} attempts "
            f"({type(problem).__name__})."
        )

    def generate_structured(self, prompt: str, **kwargs) -> dict:
        """Only reached by Semantica's second try after a valid but empty result.

        Repeating the identical request would double the cost of every passage that
        simply contains no relations, so nothing is sent.
        """
        return {}


def register() -> str:
    """Make the provider available to Semantica and return its name.

    Semantica's in-memory result cache is switched off: it would make a
    "reprocess" silently reuse earlier output and hide real token use.
    """
    provider_registry.register(PROVIDER_NAME, MeteredOpenAIProvider)
    configure_cache(enabled=False)
    return PROVIDER_NAME
