from dataclasses import dataclass
from typing import Callable

from madclean.llm.llm_clients import BaseLLMClient, OpenAIClient, GeminiClient


@dataclass(frozen=True)
class LLMSpec:
    """How to reach one model: which client to build, which model id, and which key it needs."""
    client_class: Callable[..., BaseLLMClient]
    default_model: str
    role: str
    api_key_name: str
    base_url: str | None = None


LLM_CLIENT_MAP: dict[str, LLMSpec] = {
    "Gemini": LLMSpec(
        client_class=GeminiClient,
        default_model="gemini-2.5-pro",
        role="model",
        api_key_name="GEMINI_API_KEY",
    ),
    "OpenAI": LLMSpec(
        client_class=OpenAIClient,
        default_model="gpt-5-mini",
        role="assistant",
        api_key_name="OPENAI_API_KEY",
    ),
    "Qwen_235B": LLMSpec(
        client_class=OpenAIClient,
        default_model="qwen/qwen3-235b-a22b-2507",
        role="assistant",
        api_key_name="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    ),
    "Qwen_9B": LLMSpec(
        client_class=OpenAIClient,
        default_model="qwen/qwen3.5-9b",
        role="assistant",
        api_key_name="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    ),
    "Gemma_27B": LLMSpec(
        client_class=OpenAIClient,
        default_model="google/gemma-3-27b-it",
        role="assistant",
        api_key_name="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    ),
    "Gemma_12B": LLMSpec(
        client_class=OpenAIClient,
        default_model="google/gemma-3-12b-it",
        role="assistant",
        api_key_name="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    ),
}
