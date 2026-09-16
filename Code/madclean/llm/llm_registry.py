from madclean.llm.llm_clients import OpenAIClient, GeminiClient

LLM_CLIENT_MAP = {
    "Gemini": {
        "class": GeminiClient, 
        "default_model": "gemini-2.5-pro", 
        "role": "model",
        "api_key_name": "GEMINI_API_KEY"
    }, 
    "OpenAI": {
        "class": OpenAIClient, 
        "default_model": "gpt-5-mini", 
        "role": "assistant",
        "base_url": None,
        "api_key_name": "OPENAI_API_KEY"
    },
    "Qwen_235B": {
        "class": OpenAIClient, 
        "default_model": "qwen/qwen3-235b-a22b-2507", 
        "role": "assistant",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_name": "OPENROUTER_API_KEY"
    },
    "Qwen_9B": {
        "class": OpenAIClient, 
        "default_model": "qwen/qwen3.5-9b",
        "role": "assistant",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_name": "OPENROUTER_API_KEY"
    },           
    "Gemma_27B": {
        "class": OpenAIClient, 
        "default_model": "google/gemma-3-27b-it", 
        "role": "assistant",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_name": "OPENROUTER_API_KEY"
    },
    "Gemma_12B": {
        "class": OpenAIClient, 
        "default_model": "google/gemma-3-12b-it", 
        "role": "assistant",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_name": "OPENROUTER_API_KEY"
    }
}
