import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from openai import AsyncOpenAI

from google import genai
from google.genai import types
from pydantic import BaseModel


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
    def __init__(self, model_name: str):
        self.model_name = model_name
    
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
            
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    async def call_llm_async(
        self,
        messages: list[dict],
        response_schema: Any = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> tuple[str, dict]:
        config: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "extra_body": {"enable_thinking": False},
        }

        if response_schema != "text/plain":
            config["response_format"] = response_schema

        sampling: dict[str, float] = {}
        if temperature is not None:
            sampling["temperature"] = float(temperature)
        if top_p is not None:
            sampling["top_p"] = float(top_p)

        try:
            if sampling:
                response = await self.async_client.chat.completions.create(**config, **sampling)
            else:
                response = await self.async_client.chat.completions.create(**config)
        except Exception as exc:
            if sampling and _sampling_error_retry_without(exc):
                response = await self.async_client.chat.completions.create(**config)
            else:
                raise

        token_usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
        return response.choices[0].message.content, token_usage
 
class GeminiClient(BaseLLMClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.sync_client = genai.Client()
        self.async_client = self.sync_client.aio

    async def call_llm_async(
        self,
        messages: list[dict],
        response_schema: Any = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
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
            response = await self.async_client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
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
                response = await self.async_client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
            else:
                raise
        token_usage = {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": (response.usage_metadata.total_token_count - response.usage_metadata.prompt_token_count),
        }
        return response.text, token_usage