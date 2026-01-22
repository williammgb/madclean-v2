import json
from pydantic import BaseModel
import pandas as pd
# Local imports
from macs.llm.llm_clients import BaseLLMClient
from macs.components.coordinator.prompt_generation import PromptGeneration
from macs.components.domain.schema import ColumnProfile, MultiColumnTask

class CodeOutputRecommendation(BaseModel):
    """Defines the required JSON output structure for cleaning instructions."""
    is_clean: bool
    summary: str
    error_types: list[str] | None
    examples_clean: list[str] | None
    examples_dirty: list[str] | None
    cleaning_instructions: list[str] | None

class CodeOutputFDRecommendation(BaseModel):
    """Defines the required JSON output structure for FD instructions."""
    summary: str
    violation_instructions: str
    imputation_instructions: str

# Example of future extension
# class CodeOutputAdditionalComponentRecommendation(BaseModel):
#     summary: str
#     correction_instructions: str

class LLMRecommendationAgent:
    """Isolated LLM Agent that analyses data provided by DataProfiler and generates cleaning instructions for LLMCodingAgent."""
    def __init__(self, llm_client: BaseLLMClient,  llm_role: str, update_token_func, verbose: bool = False):
        self.verbose = verbose
        self.llm_client = llm_client
        self.llm_role = llm_role
        self._update_token_func = update_token_func
        self.prompt_generator = PromptGeneration()
        self.system_prompt = (
            "You are an expert data analyst specializing in tabular dataset cleaning. "
            "Your role is to analyze a data column and provide clear cleaning instructions. "
            "You MUST respond in the specified JSON format."
        )
        self.multi_col_config = {
            'FD': {
                'schema': CodeOutputFDRecommendation,
                'system_prompt' : (
                    "You are an expert data analyst specializing in enforcing functional dependencies (FDs) in tabular data. "
                    "Your role is to analyze an FD and provide clear instructions for resolving violations and imputing missing values. "
                    "You MUST respond in the specified JSON format.")
            }
            # Example of future extension
            # 'AdditionalComponent': {
            #     'schema': CodeOutputAdditionalComponentRecommendation,
            #     'system_prompt': "You are an expert ...."
            # }
        }  

    async def generate_recommendations_async(self, col: str, column_profile: ColumnProfile, messages: list[dict[str, str]] | None = None) -> tuple[bool, dict | None, list | None]:
        """Creates prompt with given data and passes it to LLM to generate column cleaning instructions."""
        # if self.verbose: print(f"[{col}] Recommender Agent: analyzing column...")
        if messages is None:
            # 1. First time the recommender agent is called. Provide profiler data.
            messages = [{"role": "system", "content": self.system_prompt}] 
            initial_prompt = self.prompt_generator.create_prompt_recommender(col, column_profile)     
            messages.append({"role": "user", "content": initial_prompt})
        # 2. Call LLM to generate instructions
        raw_response, token_usage = await self.llm_client.call_llm_async(messages, response_schema=CodeOutputRecommendation)
        await self._track_usage(token_usage)
        messages.append({"role": self.llm_role, "content": raw_response})
        # 3. Parse output
        response = json.loads(raw_response)
        is_clean = response.get('is_clean')
        instructions = response.get('cleaning_instructions') 
        if is_clean:
            # if self.verbose: print(f"[{col}] Recommender Agent: Column is already clean")
            return True, None, None
        if not instructions:
            # if self.verbose: print(f"[{col}] Recommender Agent: No instructions provided.")
            return False, None, messages
        # if self.verbose: print(f"[{col}] Recommender Agent: Succesfully generated instructions.")
        return False, response, messages

    async def generate_recommendations_multi_col_async(self, df: pd.DataFrame, task_info: MultiColumnTask, messages: list[dict[str, str]] | None = None) -> tuple[dict | None, list | None]:
        """Creates prompt with given data and passes it to LLM to generate multi-column cleaning instructions."""
        task_type = task_info.task_type
        task_key = task_info.verbose_key
        # 1. Look-up configuration
        config = self.multi_col_config.get(task_type)
        if not config:
            raise ValueError(f"Recommender Agent does not support task type: '{task_type}'")
        target_schema = config['schema']
        system_prompt = config['system_prompt']
        # if self.verbose: print(f"[{task_key}] Recommender Agent: analyzing {task_type}...")
        if messages is None:
            # 1. First time the recommender agent is called. Provide multi-column data.
            messages = [{"role": "system", "content": system_prompt}]
            initial_prompt = self.prompt_generator.create_prompt_recommender_multi_col(df, task_info)
            messages.append({"role": "user", "content": initial_prompt})
        # 2. Call LLM to generate instructions
        raw_response, token_usage = await self.llm_client.call_llm_async(messages, response_schema=target_schema)
        await self._track_usage(token_usage)
        messages.append({"role": self.llm_role, "content": raw_response})
        # 3. Parse output, works for FDs and future extension schemas
        response = json.loads(raw_response)
        has_content = False
        for key, value in response.items():
            if key != 'summary' and value:
                has_content = True
                break
        if not has_content:
            if self.verbose: print(f"[{task_key}] Recommender Agent: No instructions provided.")
            return None, messages
        # if self.verbose: print(f"[{task_key}] Recommender Agent: Succesfully generated instructions.")
        return response, messages
    
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
#     from macs.llm.llm_settings import LLM_CLIENT_NAME
#     from macs.llm.llm_registry import LLM_CLIENT_MAP

#     llm_client_name = LLM_CLIENT_NAME
#     llm_clients = LLM_CLIENT_MAP
#     llm_client = llm_clients[llm_client_name]
#     api_key_name = llm_client["api_key_name"]
#     api_key = os.getenv(api_key_name)
#     client = llm_client["class"](model_name=llm_client["default_model"])
#     llm_role=llm_client["role"]
    
#     async def mock_update_token_count(input_tokens: int, output_tokens: int):
#         print(f"Input: {input_tokens} | Output: {output_tokens}")
#     recommender_agent = LLMRecommendationAgent(client, llm_role, mock_update_token_count, verbose=True)
    import asyncio
    from macs.components.domain.schema import OutlierResult, FDResult
    class MockLLMClient:
        async def call_llm_async(self, messages, response_schema):
            # Mock response based on schema type
            if response_schema == CodeOutputRecommendation:
                return json.dumps({
                    "is_clean": False, "summary": "Test Summary", 
                    "cleaning_instructions": ["Do this", "Do that"]
                }), {"input_tokens": 10, "output_tokens": 10}
            elif response_schema == CodeOutputFDRecommendation:
                return json.dumps({
                    "summary": "FD Analysis", 
                    "violation_instructions": "Fix 'wrong1' to 'z'", 
                    "imputation_instructions": "Impute 'null' with 'x'"
                }), {"input_tokens": 15, "output_tokens": 15}
            return "{}", {}

    async def mock_update_token_count(input_tokens: int, output_tokens: int):
        print(f"Input: {input_tokens} | Output: {output_tokens}")

    # Initialize Agent
    mock_client = MockLLMClient()
    recommender_agent = LLMRecommendationAgent(mock_client, "assistant", mock_update_token_count, verbose=True)
  
    async def run_test1():
        print("\n==== TESTING SINGLE COLUMN CLEANING ====")
        test_col = "Age"
        outlier_obj = OutlierResult(
            median=29,
            mad=5,
            outliers=[(150, 1), (-5, 1)],
            context=[] 
        )
        test_profile = ColumnProfile(
            name=test_col,
            semantic_type="INTEGER",
            sample="25, 32, 28, 150, -5, null, 30",
            outlier_data=outlier_obj
        )
        is_clean, response, history = await recommender_agent.generate_recommendations_async(col=test_col, column_profile=test_profile)
        if response:
            print(f"Summary: {response.get('summary')}")
            print(f"Cleaning Instructions: {response.get('cleaning_instructions')}")
        else:
            print("No response generated")

    async def run_test2():
        print("\n==== TESTING FD CLEANING ====")
        df = pd.DataFrame({
            "col1": ["A", "A", "B", "B", "C", "C", "C", "C", "E"],
            "col2": ["x", None, "y", "y", "z", "wrong1", "z", "wrong2", "m"],
            "col3": ["foo", "bar", "lmn", "xyz", "aaa", "bbb", "ccc", "ddd", "eee"]
        })
        violation_payload = {
            'count': 1, 
            'violations': [{
                'lhs': 'C', 
                'rhs_conflicts': [('z', 2), ('wrong1', 1), ('wrong2', 1)], 
                'context': [['C', 'z', 'aaa'], ['C', 'wrong1', 'bbb'], ['C', 'wrong2', 'ddd']]
            }]
        }
        fd_result = FDResult(
            lhs='col1',
            rhs='col2',
            score=1.0,
            violations_count=1,
            imputables_count=1,
            violation_data=violation_payload,
            imputation_data={'count': 1}
        )
        task_info = MultiColumnTask(
            task_type='FD',
            target_columns=['col1', 'col2'],
            verbose_key='col1 -> col2',
            data=fd_result
        )

        recommender_data, history = await recommender_agent.generate_recommendations_multi_col_async(df, task_info)
        if recommender_data:
            print("Response received:")
        else:
            print('No reponse received')

    async def main():
        await run_test1()
        await run_test2()
     
    asyncio.run(main())
    # python -m macs.components.multi_agent_cleaner.llm_recommending