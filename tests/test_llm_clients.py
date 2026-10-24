import asyncio
import json
from types import SimpleNamespace

import pytest

from madclean.components.multi_agent_cleaner.llm_validation import CodeOutputValidation
from google.genai import errors as genai_errors

from madclean.llm.llm_clients import GeminiClient, OpenAIClient
from madclean.llm.llm_registry import LLM_CLIENT_MAP

MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


class _StubCompletions:
    """Records how the client calls the OpenAI SDK and answers like it."""

    def __init__(self, parse_error: Exception | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.parse_error = parse_error

    @staticmethod
    def _response(content=None, parsed=None):
        message = SimpleNamespace(content=content, parsed=parsed)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )

    async def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return self._response(content='{"from": "create"}')

    async def parse(self, **kwargs):
        self.calls.append(("parse", kwargs))
        if self.parse_error:
            raise self.parse_error
        parsed = CodeOutputValidation(needs_correction=False, feedback_target=None, correction_instructions="")
        return self._response(parsed=parsed)


def _client(base_url=None, parse_error=None):
    client = OpenAIClient(model_name="m", api_key="test", base_url=base_url)
    completions = _StubCompletions(parse_error)
    client.async_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_openai_with_a_schema_uses_parse_without_extra_body():
    client, completions = _client()

    content, usage = asyncio.run(client.call_llm_async(MESSAGES, response_schema=CodeOutputValidation))

    (method, kwargs), = completions.calls
    assert method == "parse"
    assert kwargs["response_format"] is CodeOutputValidation
    assert "extra_body" not in kwargs
    assert json.loads(content) == {"needs_correction": False, "feedback_target": None, "correction_instructions": ""}
    assert usage == {"input_tokens": 11, "output_tokens": 7}


def test_openrouter_with_a_schema_asks_for_a_json_object_and_keeps_extra_body():
    client, completions = _client(base_url="https://openrouter.ai/api/v1")

    content, _ = asyncio.run(client.call_llm_async(MESSAGES, response_schema=CodeOutputValidation))

    (method, kwargs), = completions.calls
    assert method == "create"
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["extra_body"] == {"reasoning": {"enabled": False}}
    assert content == '{"from": "create"}'


def test_plain_text_requests_send_no_response_format():
    client, completions = _client()

    asyncio.run(client.call_llm_async(MESSAGES, response_schema="text/plain"))

    (method, kwargs), = completions.calls
    assert method == "create"
    assert "response_format" not in kwargs


def test_plain_text_requests_to_openai_send_no_openrouter_setting():
    client, completions = _client()

    asyncio.run(client.call_llm_async(MESSAGES, response_schema="text/plain"))

    (_, kwargs), = completions.calls
    assert "extra_body" not in kwargs


def test_plain_text_requests_to_openrouter_keep_its_setting():
    client, completions = _client(base_url="https://openrouter.ai/api/v1")

    asyncio.run(client.call_llm_async(MESSAGES, response_schema="text/plain"))

    (_, kwargs), = completions.calls
    assert kwargs["extra_body"] == {"reasoning": {"enabled": False}}


class _HangingCompletions(_StubCompletions):
    """Never replies to the first `hangs` requests, then answers normally."""

    def __init__(self, hangs: int):
        super().__init__()
        self.hangs = hangs

    async def create(self, **kwargs):
        if len(self.calls) < self.hangs:
            self.calls.append(("create", kwargs))
            await asyncio.sleep(3600)
        return await super().create(**kwargs)


def _hanging_client(hangs):
    client, _ = _client(base_url="https://openrouter.ai/api/v1")
    completions = _HangingCompletions(hangs)
    client.async_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client.timeout_seconds, client.attempts, client.retry_wait_seconds = 0.05, 3, 0
    return client, completions


def test_a_reply_that_hangs_once_is_asked_for_again():
    client, completions = _hanging_client(hangs=1)

    content, _ = asyncio.run(client.call_llm_async(MESSAGES, response_schema="text/plain"))

    assert len(completions.calls) == 2
    assert content == '{"from": "create"}'


def test_a_reply_that_never_comes_gives_up_after_three_tries():
    client, completions = _hanging_client(hangs=99)

    with pytest.raises(TimeoutError, match="no reply within 0.05 seconds"):
        asyncio.run(client.call_llm_async(MESSAGES, response_schema="text/plain"))

    assert len(completions.calls) == 3


def test_unparseable_structured_output_falls_back_to_a_json_object_request():
    client, completions = _client(parse_error=json.JSONDecodeError("bad", "doc", 0))

    content, _ = asyncio.run(client.call_llm_async(MESSAGES, response_schema=CodeOutputValidation))

    assert [method for method, _ in completions.calls] == ["parse", "create"]
    assert completions.calls[1][1]["response_format"] == {"type": "json_object"}
    assert content == '{"from": "create"}'


def test_other_parse_errors_are_raised():
    client, _ = _client(parse_error=RuntimeError("provider down"))

    with pytest.raises(RuntimeError, match="provider down"):
        asyncio.run(client.call_llm_async(MESSAGES, response_schema=CodeOutputValidation))


def test_qwen_9b_uses_the_real_openrouter_model_id():
    assert LLM_CLIENT_MAP["Qwen_9B"].default_model == "qwen/qwen3.5-9b"


class _StubGeminiModels:
    """Plays one scripted outcome per call: "hang", an exception, or a reply."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def generate_content(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if outcome == "hang":
            await asyncio.sleep(3600)
        if isinstance(outcome, Exception):
            raise outcome
        usage = SimpleNamespace(prompt_token_count=10, total_token_count=14)
        return SimpleNamespace(text=outcome, usage_metadata=usage)


def _gemini(outcomes):
    client = GeminiClient.__new__(GeminiClient)
    client.model_name = "gemini-test"
    client.timeout_seconds = 0.05
    client.retry_wait_seconds = 0
    models = _StubGeminiModels(outcomes)
    client.async_client = SimpleNamespace(models=models)
    return client, models


def test_gemini_retries_a_call_that_never_answers():
    client, models = _gemini(["hang", "ok"])

    content, usage = asyncio.run(client.call_llm_async(MESSAGES, response_schema="text/plain"))

    assert (content, usage) == ("ok", {"input_tokens": 10, "output_tokens": 4})
    assert models.calls == 2


def test_gemini_gives_up_with_a_clear_error_after_three_hangs():
    client, models = _gemini(["hang", "hang", "hang"])

    with pytest.raises(TimeoutError, match="no reply within 0.05 seconds .try 3 of 3"):
        asyncio.run(client.call_llm_async(MESSAGES, response_schema="text/plain"))
    assert models.calls == 3


def test_gemini_retries_a_server_fault_and_a_rate_limit():
    unavailable = genai_errors.ServerError(503, {"error": {"message": "busy", "status": "UNAVAILABLE"}})
    rate_limited = genai_errors.ClientError(429, {"error": {"message": "slow down", "status": "RESOURCE_EXHAUSTED"}})
    client, models = _gemini([unavailable, rate_limited, "ok"])

    content, _ = asyncio.run(client.call_llm_async(MESSAGES, response_schema="text/plain"))

    assert content == "ok"
    assert models.calls == 3


def test_gemini_does_not_retry_a_bad_request():
    bad = genai_errors.ClientError(400, {"error": {"message": "bad input", "status": "INVALID_ARGUMENT"}})
    client, models = _gemini([bad, "never reached"])

    with pytest.raises(genai_errors.ClientError, match="bad input"):
        asyncio.run(client.call_llm_async(MESSAGES, response_schema="text/plain"))
    assert models.calls == 1
