import os
from abc import ABC, abstractmethod
from openai import AsyncOpenAI

from google import genai
from google.genai import types
from pydantic import BaseModel

class BaseLLMClient(ABC):
    """Abstract base class specifying how LLM APIs must be implemented."""
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    async def call_llm_async(self, messages: list[dict], response_schema = None) -> tuple[str, dict]:
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
    
    async def call_llm_async(self, messages: list[dict], response_schema = None) -> tuple[str, dict]:
        config = {
                "model": self.model_name, 
                "messages": messages, 
                "extra_body": {"enable_thinking": False }
        }
        
        if response_schema != "text/plain":
            config["response_format"] = response_schema
        
        response = await self.async_client.chat.completions.create(**config)

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

    async def call_llm_async(self, messages: list[dict], response_schema = None) -> tuple[str, dict]:
        system_prompt = messages[0]["content"]
        chat_messages = messages[1:]
        contents = []
        for msg in chat_messages:
            contents.append(types.Content(role=msg["role"], parts=[types.Part(text=msg['content'])]))
        
        if response_schema == "text/plain":
            # for coding agent --> no JSON
            config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="text/plain")
        else:
            # for other agents --> return JSON
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