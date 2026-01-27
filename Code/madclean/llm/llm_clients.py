import os
from abc import ABC, abstractmethod
from openai import AsyncOpenAI
from typing import Tuple, List, Dict

from google import genai
from google.genai import types
from pydantic import BaseModel

class BaseLLMClient(ABC):
    """Abstract base class specifying how LLM APIs must be implemented."""
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    async def call_llm_async(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, int]]:
        pass

class OpenAIClient(BaseLLMClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.async_client = AsyncOpenAI()
    
    async def call_llm_async(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, int]]:
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"}
        )
        token_usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens
        }
        return response.choices[0].message.content, token_usage
    
class GeminiClient(BaseLLMClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.sync_client = genai.Client()
        self.async_client = self.sync_client.aio

    async def call_llm_async(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, int]]:
        response_schema = kwargs.get("response_schema")
        system_prompt = messages[0]["content"]
        chat_messages = messages[1:]
        
        contents = []
        for msg in chat_messages:
            contents.append(types.Content(role=msg["role"], parts=[types.Part(text=msg['content'])]))
        
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
        token_usage = {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": (response.usage_metadata.total_token_count - response.usage_metadata.prompt_token_count)
        }
        return response.text, token_usage