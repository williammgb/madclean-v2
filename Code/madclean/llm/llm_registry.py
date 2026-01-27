from madclean.llm.llm_clients import OpenAIClient, GeminiClient

LLM_CLIENT_MAP = {
    "OpenAI": {"class": OpenAIClient, 
               "default_model": "gpt-5-mini", 
               "role": "assistant",
               "api_key_name": "OPENAI_API_KEY"
               }, # 400000 context window
    "Gemini": {"class": GeminiClient, 
               "default_model": "gemini-2.5-pro", 
               "role": "model",
               "api_key_name": "GEMINI_API_KEY"}
                 # 1 million context window
}
