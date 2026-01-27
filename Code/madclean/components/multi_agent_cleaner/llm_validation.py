import json
import pandas as pd
from pydantic import BaseModel
from typing import Literal
# Local imports
from madclean.llm.llm_clients import BaseLLMClient
from madclean.components.coordinator.prompt_generation import PromptGeneration
from madclean.components.domain.schema import MultiColumnTask

class CodeOutputValidation(BaseModel):
    """Defines the required JSON output strcuture for verification of Gemini API."""
    needs_correction: bool
    feedback_target: Literal["CODER", "RECOMMENDER"] | None
    correction_instructions: str

class LLMValidationAgent:
    """Isolated LLM agent that validates the performed cleaning operations, gives feedback if operations are incorrect."""
    def __init__(self, llm_client: BaseLLMClient, llm_role: str, update_token_func, verbose: bool = False):
        self.verbose = verbose
        self.llm_client = llm_client
        self.llm_role = llm_role
        self.system_prompt = "You are a data validation expert tasked with evaluating if a cleaned dataset remains consistent with the original data, detecting unwanted or excessive transformations."
        self._update_token_func = update_token_func
        self.prompt_generator = PromptGeneration()
        
    async def validate_async(self, 
               col: str, 
               dirty_column: pd.Series, 
               cleaned_column: pd.Series,
               column_type: str, 
               messages: list[dict[str, str]] | None = None,
               last_attempt: bool = False) -> tuple[bool, str | None, str| None, list[dict[str, str]]]: 
        """
        Compares dirty and cleaned columns and validates cleaning operations.
        If validator detects undesired changes, it sends feedback to the corresponding LLM agent.
        """
        # if self.verbose: print(f"[{col}] Validating cleaning...")
        # 1. Using validator for INTEGER, FLOAT and BOOLEAN types is unnecessary
        if column_type in ("INTEGER", "FLOAT", "BOOLEAN"):
            return False, None, None, messages 
        # 2. Create validation prompt
        if messages is None:
            messages = [{"role": "system", "content": self.system_prompt}]
        prompt = self.prompt_generator.create_prompt_validation(col, dirty_column, cleaned_column, column_type, last_attempt)
        messages.append({"role": "user", "content": prompt})
        # 3. Call Validation Agent to validate cleaning operations
        raw_response, token_usage = await self.llm_client.call_llm_async(messages, response_schema=CodeOutputValidation)
        await self._track_usage(token_usage)
        messages.append({"role": self.llm_role, "content": raw_response})
        # 4. Parse output
        response = json.loads(raw_response)
        needs_correction = response.get("needs_correction")
        if isinstance(needs_correction, str):
            needs_correction = needs_correction.strip().lower() == 'true'
        feedback_target = response.get("feedback_target", None) # str | None
        correction_instructions = response.get("correction_instructions", "")
        # 5. If validation failed, send feedback. Otherwise accept cleaned column
        if needs_correction and correction_instructions:
            # if self.verbose: print(f"[{col}] Validation FAILED! -> Feedback for {feedback_target}: {correction_instructions}")
            return True, feedback_target, correction_instructions, messages
        else:
            # if self.verbose: print(f"[{col}] Validation PASSED.")
            return False, None, None, messages

    async def validate_multi_col_async(self,
                dirty_targets: pd.DataFrame,
                cleaned_targets: pd.DataFrame,
                task_info: MultiColumnTask,
                messages: list[dict[str, str]] | None = None,
                last_attempt: bool = False) -> tuple[bool, str | None, str | None, list[dict[str, str]]]:
        """
        Compares dirty and cleaned column pairs and validates cleaning operations.
        If validator detects undesired changes, it sends feedback to the corresponding LLM agent.
        """
        task_key = task_info.verbose_key
        # if self.verbose: print(f"[{task_key}] Validating cleaning...")        
        # 1. Create validation prompt
        if messages is None:
            messages = [{"role": "system", "content": self.system_prompt}]
        prompt = self.prompt_generator.create_prompt_validation_multi_col(dirty_targets, cleaned_targets, task_info, last_attempt)        
        messages.append({"role": "user", "content": prompt})
        # 2. Call Validation Agent
        raw_response, token_usage = await self.llm_client.call_llm_async(messages, response_schema=CodeOutputValidation)
        await self._track_usage(token_usage)
        messages.append({"role": self.llm_role, "content": raw_response})
        # 3. Parse output
        response = json.loads(raw_response)
        needs_correction = response.get("needs_correction")
        if isinstance(needs_correction, str):
            needs_correction = needs_correction.strip().lower() == 'true'
        feedback_target = response.get("feedback_target", None) 
        correction_instructions = response.get("correction_instructions", "")
        # 5. If validation failed, send feedback. Otherwise accept cleaned column
        if needs_correction and correction_instructions:
            # if self.verbose: print(f"[{task_key}] Validation FAILED! -> Feedback for {feedback_target}: {correction_instructions}") 
            return True, feedback_target, correction_instructions, messages
        else:
            # if self.verbose: print(f"[{task_key}] Validation PASSED.")
            return False, None, None, messages
    
    async def _track_usage(self, token_usage):
        input_tokens = token_usage.get('input_tokens', 0)
        output_tokens = token_usage.get('output_tokens', 0)
        await self._update_token_func(input_tokens, output_tokens)

####### TEST CODE #######
if __name__ == "__main__":
#     import asyncio
#     import os
#     from dotenv import load_dotenv
#     load_dotenv()
#     from madclean.llm.llm_settings import LLM_CLIENT_NAME
#     from madclean.llm.llm_registry import LLM_CLIENT_MAP

#     llm_client_name: str = LLM_CLIENT_NAME
#     llm_clients: dict = LLM_CLIENT_MAP
#     llm_client = llm_clients[llm_client_name]
#     api_key_name = llm_client["api_key_name"]
#     api_key = os.getenv(api_key_name)
#     client = llm_client["class"](model_name=llm_client["default_model"])
#     llm_role=llm_client["role"]

#     async def mock_update_token_count(input_tokens: int, output_tokens: int):
#         print(f"   [Token Update] Input: {input_tokens} | Output: {output_tokens}")
#     validation_agent = LLMValidationAgent(client, llm_role, mock_update_token_count, verbose=True)
    import asyncio
    import json
    import pandas as pd
    from madclean.components.domain.schema import FDResult   

    class MockLLMClient:
        async def call_llm_async(self, messages, response_schema):
            messages_str = str(messages).lower()
            if "city" in messages_str and "san francisco" in messages_str:
                return json.dumps({
                    "needs_correction": True,
                    "feedback_target": "CODER",
                    "correction_instructions": (
                        "The cleaning was too aggressive. "
                        "The entity 'San Francisco' was truncated to 'san'. "
                        "Please retain the full city name and only remove whitespace."
                    )
                }), {"input_tokens": 150, "output_tokens": 60}

            if "violations" in messages_str:
                return json.dumps({
                    "needs_correction": False,
                    "feedback_target": None,
                    "correction_instructions": ""
                }), {"input_tokens": 200, "output_tokens": 10}

            # return json.dumps({
            #     "needs_correction": False,
            #     "feedback_target": None,
            #     "correction_instructions": ""
            # }), {"input_tokens": 50, "output_tokens": 10}
    
    async def mock_update_token_count(input_tokens: int, output_tokens: int):
        print(f"Input: {input_tokens} | Output: {output_tokens}")
    
    mock_client = MockLLMClient()
    validation_agent = LLMValidationAgent(mock_client, "assistant", mock_update_token_count, verbose=True)

    async def run_test1():
        print("\n==== TESTING SINGLE COLUMN CLEANING ====")
        col_name = "City"
        column_type = "NAMED_ENTITY"
        dirty_list = [
            " San Francisco ", 
            "san francisco", 
            "New York", 
            "unknown",       
            "Los Angeles"
        ]
        dirty_series = pd.Series(dirty_list, name=col_name)
        cleaned_list = [
            "san",           
            "san",           
            "new york",      
            "unknown",       
            "los"        
        ]
        cleaned_series = pd.Series(cleaned_list, name=col_name)
        needs_correction, feedback_target, correction_instructions, new_validator_history = await validation_agent.validate_async(
                col_name, dirty_series, cleaned_series, column_type)
        if needs_correction:
            print(f"{correction_instructions}")
        else:
            print('Validation PASSED. (should not happen)')

    async def run_test2():
        print("\n==== TESTING MULTI COLUMN CLEANING ====")
        df_dirty = pd.DataFrame({
            "A": ["1", "1", "2"],
            "B": ["x", "y", "z"] 
        })
        df_clean = pd.DataFrame({
            "A": ["1", "1", "2"],
            "B": ["x", "x", "z"] 
        })
        task_info = MultiColumnTask(
            task_type='FD',
            target_columns=['A', 'B'],
            verbose_key='A -> B',
            data=FDResult(
                lhs='A',
                rhs='B',
                score=1.0
            )
        )
        needs_correction, target, instructions, history = await validation_agent.validate_multi_col_async(
            dirty_targets=df_dirty,
            cleaned_targets=df_clean,
            task_info=task_info
        )
        if not needs_correction:
            print(f"\n[Result] Validation PASSED. (Which should be the case.)")
        else:
            print(f"\n[Result] Validation FAILED with feedback: {instructions}")

    async def main():
        await run_test1()
        await run_test2()

    asyncio.run(main())

    # python -m madclean.components.multi_agent_cleaner.llm_validation