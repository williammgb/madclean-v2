import asyncio
import json
from types import SimpleNamespace

import pytest

from madclean.components.multi_agent_cleaner.llm_validation import CodeOutputValidation
from madclean.llm.llm_clients import OpenAIClient
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
    assert kwargs["extra_body"] == {"enable_thinking": False}
    assert content == '{"from": "create"}'


def test_plain_text_requests_send_no_response_format():
    client, completions = _client()

    asyncio.run(client.call_llm_async(MESSAGES, response_schema="text/plain"))

    (method, kwargs), = completions.calls
    assert method == "create"
    assert "response_format" not in kwargs


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
    assert LLM_CLIENT_MAP["Qwen_9B"]["default_model"] == "qwen/qwen3.5-9b"
