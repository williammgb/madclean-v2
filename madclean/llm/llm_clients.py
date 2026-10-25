import asyncio
import json
import os
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Any, Optional

from openai import AsyncOpenAI

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from madclean.llm.activity import current_agent, current_attempt, current_task

# The id of the call this asyncio task has open in the activity record, for the retry loop to report on.
_open_call_id: ContextVar[int | None] = ContextVar("madclean_open_call_id", default=None)


def _gemini_retry_reason(exc: BaseException) -> str:
    """Which of the three passing faults made the call go again, in words for the screen."""
    if isinstance(exc, TimeoutError):
        return "timed out"
    if isinstance(exc, genai_errors.ServerError):
        return "server error"
    return "rate limited"


def _gemini_error_is_worth_retrying(exc: BaseException) -> bool:
    """A reply that never came, a server fault or a rate limit may pass on a second try."""
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, genai_errors.ServerError):
        return True
    return isinstance(exc, genai_errors.APIError) and getattr(exc, "code", None) == 429


def _sampling_error_retry_without(exc: BaseException) -> bool:
    """Some models reject temperature/top_p; retry without sampling overrides."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "temperature" in msg or "top_p" in msg or "top p" in msg or "sampling" in msg:
        return True
    code = getattr(exc, "status_code", None)
    if code is None and getattr(exc, "response", None) is not None:
        code = getattr(exc.response, "status_code", None)
    return code == 400


class BaseLLMClient(ABC):
    """Abstract base class specifying how LLM APIs must be implemented."""
    # The Pipeline's live record of open calls; None (tests, CLI) records nothing.
    activity = None

    def __init__(self, model_name: str):
        self.model_name = model_name

    def _activity_start(self) -> int | None:
        """Record a call as open, for the agent, task and attempt of this asyncio task."""
        if self.activity is None:
            return None
        return self.activity.start(
            current_agent.get(), current_task.get(), current_attempt.get(), self.model_name
        )

    def _activity_finish(self, call_id: int | None) -> None:
        if self.activity is not None and call_id is not None:
            self.activity.finish(call_id)

    def _activity_retrying(self, try_no: int, tries: int, reason: str) -> None:
        call_id = _open_call_id.get()
        if self.activity is not None and call_id is not None:
            self.activity.retrying(call_id, try_no, tries, reason)

    async def _reporting(self, call):
        """Run one call coroutine with it recorded as open, and closed again however it ends."""
        call_id = self._activity_start()
        token = _open_call_id.set(call_id)
        try:
            return await call
        finally:
            _open_call_id.reset(token)
            self._activity_finish(call_id)

    @abstractmethod
    async def call_llm_async(
        self,
        messages: list[dict],
        response_schema: Any = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> tuple[str, dict]:
        pass

class OpenAIClient(BaseLLMClient):
    def __init__(
        self, 
        model_name: str, 
        api_key: str = None, 
        base_url: str = None
    ):
    
        super().__init__(model_name)

        if not api_key:
            if base_url and "openrouter.ai" in base_url:
                api_key = os.getenv("OPENROUTER_API_KEY")
            else:
                api_key = os.getenv("OPENAI_API_KEY")

        self.base_url = base_url
        self._is_openrouter = bool(base_url and "openrouter.ai" in base_url)
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )

    async def _openai_create(
        self,
        config: dict[str, Any],
        sampling: dict[str, float],
    ):
        if sampling:
            return await self.async_client.chat.completions.create(**config, **sampling)
        return await self.async_client.chat.completions.create(**config)

    async def _openai_parse(
        self,
        config: dict[str, Any],
        response_schema: type[BaseModel],
        sampling: dict[str, float],
    ):
        if sampling:
            return await self.async_client.chat.completions.parse(
                **config, response_format=response_schema, **sampling
            )
        return await self.async_client.chat.completions.parse(
            **config, response_format=response_schema
        )

    async def call_llm_async(
        self,
        messages: list[dict],
        response_schema: Any = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> tuple[str, dict]:
        return await self._reporting(self._call(messages, response_schema, temperature, top_p))

    async def _call(
        self,
        messages: list[dict],
        response_schema: Any,
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> tuple[str, dict]:
        config: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "extra_body": {"enable_thinking": False},
        }

        use_pydantic_schema = (
            response_schema != "text/plain"
            and isinstance(response_schema, type)
            and issubclass(response_schema, BaseModel)
        )
        # OpenRouter does not reliably support chat.completions.parse().
        use_parse = use_pydantic_schema and not self._is_openrouter
        if use_parse:
            config.pop("extra_body", None)
        elif use_pydantic_schema:
            config["response_format"] = {"type": "json_object"}
        elif response_schema != "text/plain":
            config["response_format"] = response_schema

        sampling: dict[str, float] = {}
        if temperature is not None:
            sampling["temperature"] = float(temperature)
        if top_p is not None:
            sampling["top_p"] = float(top_p)

        try:
            if use_parse:
                response = await self._openai_parse(config, response_schema, sampling)
            else:
                response = await self._openai_create(config, sampling)
        except Exception as exc:
            if sampling and _sampling_error_retry_without(exc):
                try:
                    if use_parse:
                        response = await self._openai_parse(config, response_schema, {})
                    else:
                        response = await self._openai_create(config, {})
                except Exception as retry_exc:
                    raise retry_exc from exc
            elif use_parse and isinstance(exc, json.JSONDecodeError):
                fallback_config = {k: v for k, v in config.items() if k != "response_format"}
                fallback_config["response_format"] = {"type": "json_object"}
                try:
                    response = await self._openai_create(fallback_config, sampling)
                except Exception as retry_exc:
                    raise retry_exc from exc
                use_parse = False
            else:
                raise

        token_usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
        if use_parse:
            parsed = response.choices[0].message.parsed
            content = (
                parsed.model_dump_json()
                if parsed is not None
                else (response.choices[0].message.content or "")
            )
        else:
            content = response.choices[0].message.content
        return content, token_usage
 
class GeminiClient(BaseLLMClient):
    # One reply can hang forever without a limit, freezing the whole run.
    timeout_seconds: float = 180.0
    attempts: int = 3
    retry_wait_seconds: float = 5.0

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.sync_client = genai.Client()
        self.async_client = self.sync_client.aio

    async def _generate(self, contents, config):
        """Ask Gemini once, giving up after timeout_seconds; try again on a hang or a passing fault."""
        for attempt in range(1, self.attempts + 1):
            try:
                return await asyncio.wait_for(
                    self.async_client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=config,
                    ),
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    exc = TimeoutError(
                        f"Gemini gave no reply within {self.timeout_seconds:g} seconds"
                        f" (try {attempt} of {self.attempts})"
                    )
                if attempt == self.attempts or not _gemini_error_is_worth_retrying(exc):
                    raise exc
                self._activity_retrying(attempt + 1, self.attempts, _gemini_retry_reason(exc))
                await asyncio.sleep(self.retry_wait_seconds * attempt)

    async def call_llm_async(
        self,
        messages: list[dict],
        response_schema: Any = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> tuple[str, dict]:
        return await self._reporting(self._call(messages, response_schema, temperature, top_p))

    async def _call(
        self,
        messages: list[dict],
        response_schema: Any,
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> tuple[str, dict]:
        system_prompt = messages[0]["content"]
        chat_messages = messages[1:]
        contents = []
        for msg in chat_messages:
            contents.append(types.Content(role=msg["role"], parts=[types.Part(text=msg['content'])]))

        gen_kwargs: dict[str, Any] = {}
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)

        if response_schema == "text/plain":
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="text/plain",
                **gen_kwargs,
            )
        else:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=response_schema,
                **gen_kwargs,
            )

        try:
            response = await self._generate(contents, config)
        except Exception as exc:
            if gen_kwargs and _sampling_error_retry_without(exc):
                if response_schema == "text/plain":
                    config = types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="text/plain",
                    )
                else:
                    config = types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=response_schema,
                    )
                try:
                    response = await self._generate(contents, config)
                except Exception as retry_exc:
                    raise retry_exc from exc
            else:
                raise
        token_usage = {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": (response.usage_metadata.total_token_count - response.usage_metadata.prompt_token_count),
        }
        return response.text, token_usage