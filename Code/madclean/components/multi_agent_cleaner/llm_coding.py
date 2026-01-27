import json
import tempfile
import os
import re
import sys
import pickle
import subprocess
import pandas as pd
from pydantic import BaseModel
# Local imports
from madclean.llm.llm_clients import BaseLLMClient
from madclean.components.coordinator.prompt_generation import PromptGeneration
from madclean.components.domain.schema import MultiColumnTask

class CodeOutputCoding(BaseModel):
    """Defines the required JSON output strcuture for code generation of Gemini API."""
    code: str

class LLMCodingAgent:
    """Isolated LLM agent responsible for generating and executing Python code based on instruction of LLMRecommendationAgent."""
    MAX_CODER_ATTEMPTS = 3
    # Packages that LLM can use for cleaning data, besides built-in functions
    ALLOWED_PACKAGES = [
        'numpy',
        'pandas',
        'python-dateutil',
        'rapidfuzz',
        'fuzzywuzzy'
    ]
    def __init__(self, llm_client: BaseLLMClient,  llm_role: str, update_token_func, verbose: bool = False):
        self.verbose = verbose
        self.system_prompt = "You are an expert data analyst specializing in tabular dataset cleaning. You MUST ONLY respond with a single JSON object containing 'code' (a string containing Python code)."  
        self.llm_client = llm_client
        self.llm_role = llm_role
        self._update_token_func = update_token_func
        self.prompt_generator = PromptGeneration()

    async def _call_llm_for_code_async(self, messages: list[dict[str, str]]) -> tuple[str | None, str | None, str | None]:
        """Prompts the LLM to generate code and extracts the code string."""
        try:
            # 1. Call LLM and parse response
            raw_response, token_usage = await self.llm_client.call_llm_async(messages, response_schema=CodeOutputCoding)
            await self._track_usage(token_usage)
            response = json.loads(raw_response)
            code_str = response.get('code', "")
            code_str = re.sub(r"^```[ \w]*\n|```$", "", code_str.strip(), flags=re.MULTILINE)
            if not isinstance(code_str, str):
                error_msg = "LLM response JSON structure error: 'code' field missing or not of type string."
                return None, raw_response, error_msg
            return code_str, raw_response, None
        # 2. Return error messages if invalid JSON was returned or API failed
        except json.JSONDecodeError as e:
            error_msg = f"{type(e).__name__}: {e}"
            # if self.verbose: print(f"LLM output JSON format error: {error_msg}")
            return None, raw_response, error_msg 
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            # if self.verbose: print(f"LLM API call failed: {error_msg}")
            return None, None, error_msg
        
    @staticmethod
    def _execute_code(code_str: str, df: pd.DataFrame, columns: str | list[str] , timeout: int = 30) -> pd.Series | pd.DataFrame | str:
        """
        Executes LLM-generated code to clean the column in an isolated subprocess with a timeout.
        Dual-purpose: works for both cleaning isolated Series and DataFrames, based on the type of parameter 'columns'.
        """
        # 1. Configure based on columns: single column or multi-column cleaning operation
        if isinstance(columns, str):
            input_data = df[columns]
            is_series_input = True
        elif isinstance(columns, list):
            input_data = df[columns]
            is_series_input = False
        # 2. Create temporary directory to isolate code script from the current working environment
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "temp_script.py")
            input_file = os.path.join(tmpdir, 'input.pkl')
            output_file = os.path.join(tmpdir, 'result.pkl')
            input_dict = {
                'data': input_data,
                'is_series_input': is_series_input
            }
            with open(input_file, 'wb') as f:
                pickle.dump(input_dict, f)
            input_file_safe = input_file.replace(os.sep, '/')
            output_file_safe = output_file.replace(os.sep, '/')
            # 3. Add script wrapper to import required packages, define variables and handle output
            script_wrapper = f"""
import pandas as pd
import pickle
import sys
import os

with open("{input_file_safe}", 'rb') as f:
    input_dict = pickle.load(f)

input_data = input_dict['data']
is_series_input = input_dict['is_series_input']

expected_type = pd.Series if is_series_input else pd.DataFrame
expected_type_name = "Series" if is_series_input else "DataFrame"

{code_str}

if 'clean_column' not in locals() and 'clean_column' not in globals():
    raise NameError("LLM code did not define the 'clean_column' function.")
    
result = clean_column(input_data)

if not isinstance(result, expected_type):
    raise TypeError(f"clean_column must return a pandas {{expected_type_name}}, but returned {{type(result).__name__}}.")

result.to_pickle("{output_file_safe}")
sys.exit(0)
"""         
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_wrapper)
            # 4. Execute the code in subprocess
            try:
                subprocess.run(
                    [sys.executable, script_path],
                    timeout=timeout,
                    check=True,
                    capture_output=True,
                    text=True
                )
                result_file = os.path.join(tmpdir, "result.pkl")
                with open(result_file, "rb") as f:
                    return pickle.load(f)
            except subprocess.TimeoutExpired:
                return f"Execution did not finish before {timeout} seconds timeout. Check for blocking statements."
            except subprocess.CalledProcessError as e:
                stderr_text = e.stderr.strip()
                error_msg = stderr_text.split("\n")[-1]
                return error_msg

    async def clean_column_async(self, 
                            df: pd.DataFrame, 
                            col: str, 
                            column_type: str,
                            recommender_data: dict | None = None,
                            messages: list[dict[str, str]] | None = None
                            ) -> tuple[pd.Series | None, str | None, list[dict] | None]:
        """Generates and executes cleaning code for single column. With retries for code-level errors."""
        if messages is None:
            # 1. First time coder is called. Provide instructions
            messages = [{"role": "system", "content": self.system_prompt}]
            initial_user_prompt = self.prompt_generator.create_prompt_coding(col, column_type, 
                                                                             recommender_data,
                                                                             str(self.ALLOWED_PACKAGES))
            messages.append({"role": "user", "content": initial_user_prompt})
        # 2. Call LLM to generate code. With retries for incorrectly generated code
        retries_messages = messages.copy()
        for attempt in range(1, self.MAX_CODER_ATTEMPTS + 1):
            # if self.verbose: print(f"[{col}] Coding attempt {attempt}/{self.MAX_CODER_ATTEMPTS}...")
            code_str, llm_output, llm_error_msg = await self._call_llm_for_code_async(retries_messages)
            if llm_output:
                retries_messages.append({"role": self.llm_role, "content": llm_output})
            if llm_error_msg:
                if llm_output:
                    fix_prompt = f"The previous attempt resulted in an JSON parsing error. Please regenerate the entire valid JSON object now. The error was: {llm_error_msg}"
                else:             
                    fix_prompt = f"The previous attempt resulted in an API error. Please regenerate the entire valid JSON object now. The error was: {llm_error_msg}"
                retries_messages.append({"role": "user", "content": fix_prompt})
                continue
            if not code_str:
                fix_prompt = "The previous code generation attempt was successful (valid JSON), but the 'code' field was empty. Please ensure you provide the Python code in the 'code' field this time."
                retries_messages.append({"role": "user", "content": fix_prompt})
                continue 
            # 3. Execute the LLM-generated code. If correct output type, return
            exec_result = self._execute_code(code_str, df, col)
            if isinstance(exec_result, pd.Series):
                # if self.verbose: print(f"[{col}] Success on attempt {attempt}.")
                messages.append({"role": self.llm_role, "content": llm_output})
                return exec_result, messages
            else:
                exec_error_msg = exec_result
                # if self.verbose: print(f"[{col}] Execution Failed: {exec_error_msg}. Retrying...")
                fix_prompt = (
                    f"The Python code provided in the previous step failed during execution with the following error:"
                    f"\n\n{exec_error_msg}\n\n"
                    f"Please correct the code and provide the full, fixed JSON object containing the full 'code' again.")
                retries_messages.append({"role": "user", "content": fix_prompt})
        # if self.verbose: print(f"[{col}] FAILED after {self.MAX_CODER_ATTEMPTS} attempts.")
        return None, messages
    
    async def clean_multi_col_async(self, 
                            df: pd.DataFrame, 
                            task_info: MultiColumnTask,
                            recommender_data: dict | None = None,
                            messages: list[dict[str, str]] | None = None
                            ) -> tuple[pd.DataFrame | None, list[dict] | None]:
        """Generates and executes cleaning code for multi-column operations. With retries for code-level errors."""
        target_cols = task_info.target_columns
        task_key = task_info.verbose_key
        if messages is None:
            # 1. First time coder is called. Provide instructions
            messages = [{"role": "system", "content": self.system_prompt}]
            initial_prompt = self.prompt_generator.create_prompt_coding_multi_col(
                task_info, recommender_data, str(self.ALLOWED_PACKAGES))
            messages.append({"role": "user", "content": initial_prompt})
        # 2. Call LLM to generate code. With retries for incorrectly generated code
        retries_messages = messages.copy()
        for attempt in range(1, self.MAX_CODER_ATTEMPTS + 1):
            # if self.verbose: print(f"[{task_key}] Coding attempt {attempt}/{self.MAX_CODER_ATTEMPTS}...")
            code_str, llm_output, llm_error_msg = await self._call_llm_for_code_async(retries_messages)
            if llm_output:
                retries_messages.append({"role": self.llm_role, "content": llm_output})
            if llm_error_msg:
                if llm_output:
                    fix_prompt = f"The previous attempt resulted in an JSON parsing error. Please regenerate the entire valid JSON object now. The error was: {llm_error_msg}"
                else:             
                    fix_prompt = f"The previous attempt resulted in an API error. Please regenerate the entire valid JSON object now. The error was: {llm_error_msg}"
                retries_messages.append({"role": "user", "content": fix_prompt})
                continue
            if not code_str:
                fix_prompt = "The previous code generation attempt was successful (valid JSON), but the 'code' field was empty. Please ensure you provide the Python code in the 'code' field this time."
                retries_messages.append({"role": "user", "content": fix_prompt})
                continue 
            # 3. Execute the LLM-generated code. If result is correct output type, return it
            exec_result = self._execute_code(code_str, df, columns=target_cols)
            if isinstance(exec_result, pd.DataFrame):
                # if self.verbose: print(f"[{task_key}] Success on attempt {attempt}.")
                messages.append({"role": self.llm_role, "content": llm_output})
                return exec_result, messages
            else:
                exec_error_msg = exec_result
                # if self.verbose: print(f"[{task_key}] Execution Failed: {exec_error_msg}. Retrying...")
                fix_prompt = (
                    f"The Python code provided in the previous step failed during execution with the following error:"
                    f"\n\n{exec_error_msg}\n\n"
                    f"Please correct the code and provide the full, fixed JSON object containing the full 'code' again.")
                retries_messages.append({"role": "user", "content": fix_prompt})
        # if self.verbose: print(f"[{task_key}] FAILED after {self.MAX_CODER_ATTEMPTS} attempts.")
        return None, messages
    
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
#     coding_agent = LLMCodingAgent(client, llm_role, mock_update_token_count, verbose=True)
    import asyncio
    import json
    import pandas as pd
    import numpy as np
    from madclean.components.domain.schema import MultiColumnTask, FDResult
    class MockLLMClient:
        async def call_llm_async(self, messages, response_schema):
            messages_str = str(messages)
            if "Price" in messages_str or "currency" in messages_str:
                python_code = (
                    "def clean_column(data):\n"
                    "    # Convert to string, remove symbols, convert to numeric\n"
                    "    clean = data.astype(str).str.replace('$', '', regex=False)\n"
                    "    clean = clean.str.replace(' USD', '', regex=False)\n"
                    "    clean = clean.replace('Not Available', float('nan'))\n"
                    "    return pd.to_numeric(clean, errors='coerce')"
                )
                response_json = json.dumps({"code": python_code})
                return response_json, {"input_tokens": 50, "output_tokens": 50}
            elif "col1" in messages_str and "col2" in messages_str:
                python_code = (
                    "def clean_column(df):\n"
                    "    # Fix violations: col1='C' -> col2='z'\n"
                    "    mask_c = df['col1'] == 'C'\n"
                    "    df.loc[mask_c, 'col2'] = 'z'\n"
                    "    # Imputation: col1='A' -> col2='x'\n"
                    "    mask_a = (df['col1'] == 'A') & (df['col2'].isna())\n"
                    "    df.loc[mask_a, 'col2'] = 'x'\n"
                    "    return df"
                )
                response_json = json.dumps({"code": python_code})
                return response_json, {"input_tokens": 80, "output_tokens": 80}
            return json.dumps({"code": "def clean_column(d): return d"}), {}
    
    async def mock_update_token_count(input_tokens: int, output_tokens: int):
        print(f"   [Token Update] Input: {input_tokens} | Output: {output_tokens}")
    
    mock_client = MockLLMClient()
    coding_agent = LLMCodingAgent(mock_client, "assistant", mock_update_token_count, verbose=True)

    async def run_test1():
        col_name = "Price"
        df_test = pd.DataFrame({
            "Product": ["A", "B", "C", "D", "E", "F"],
            "Price": ["$100", "250 USD", "300", "Not Available", "50", "100"]
        })
        recommender_data_mock = {
            "is_clean": False,
            "summary": "The column contains mixed currency formats (symbols and suffixes) and non-numeric placeholders.",
            "error_types": ["Mixed Currency Symbols", "Non-numeric placeholders"],
            "examples_clean": ["100", "50", "300"],
            "examples_dirty": ["$100", "250 USD", "Not Available"],
            "cleaning_instructions": [
                "Remove currency symbols like '$' from the beginning of strings.",
                "Remove currency suffixes like 'USD' from the end of strings.",
                "Convert values like 'Not Available' to standard NaN (np.nan).",
                "Convert the final cleaned column to Integer type."
            ]
        }
        cleaned_series, history = await coding_agent.clean_column_async(
                df=df_test,
                col=col_name,
                column_type="INTEGER",
                recommender_data=recommender_data_mock
            )
        if cleaned_series is not None:
            print(cleaned_series)
        else:
            print("Execution failed.")

    async def run_test2():
        print("\n==== TESTING FD CLEANING ====")
        df_test = pd.DataFrame({
            "col1": ["A", "A", "B", "B", "C", "C", "C", "C", "E"],
            "col2": ["x", None, "y", "y", "z", "wrong1", "z", "wrong2", "m"],
            "col3": ["foo", "bar", "lmn", "xyz", "aaa", "bbb", "ccc", "ddd", "eee"]
        })
        fd_result = FDResult(
            lhs='col1',
            rhs='col2',
            score=1.0,
            violations_count=0,
            imputables_count=0,
            violation_data={},
            imputation_data={}
        )
        task_info = MultiColumnTask(
            task_type='FD',
            target_columns=['col1', 'col2'],
            verbose_key='col1 -> col2',
            data=fd_result
        )
        recommender_data_fd = {
                "summary": "The functional dependency 'col1' -> 'col2' is violated by the value 'C' in 'col1', which maps to multiple values ('z', 'wrong1', 'wrong2'). Additionally, 'col1' value 'A' has a missing value in 'col2'.",
                "violation_instructions": "Resolve violations for 'col1' value 'C'. Based on majority occurrence, set 'col2' to 'z' where 'col1' is 'C'. Specifically replace 'wrong1' and 'wrong2' with 'z'.",
                "imputation_instructions": "Impute missing values based on existing mappings. For 'col1' value 'A', the known mapping is 'x'. Fill the missing 'col2' value with 'x' where 'col1' is 'A'."
            }
        cleaned_df, history = await coding_agent.clean_multi_col_async(
            df=df_test,
            task_info=task_info,
            recommender_data=recommender_data_fd
        )
        if cleaned_df is not None:
            print(cleaned_df)
        else:
            print("Execution failed")
    
    async def main():
        await run_test1()
        await run_test2()

    asyncio.run(main())
    # python -m madclean.components.multi_agent_cleaner.llm_coding

  




