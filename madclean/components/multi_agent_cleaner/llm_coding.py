import asyncio
import tempfile
import os
import re
import sys
import pickle
import subprocess
import pandas as pd
# Local imports
from madclean.llm.llm_clients import BaseLLMClient
from madclean.components.coordinator.prompt_generation import PromptGeneration
from madclean.components.domain.schema import MultiColumnTask
from madclean.config.settings import CleaningConfig
from madclean.utils.helpers import llm_sampling_kwargs_from_config


def _service_unavailable_coder_message(exc: BaseException) -> str | None:
    """Map HTTP 503 (and similar upstream overload) to a user-facing coder error message."""
    code = getattr(exc, "status_code", None)
    if code is None:
        resp = getattr(exc, "response", None)
        if resp is not None:
            code = getattr(resp, "status_code", None)
    if code != 503:
        return None
    return (
        "API error (HTTP 503 Service Unavailable): the LLM provider is temporarily unavailable. "
        "This is not a failure of the cleaning system. System will retry in a few seconds."
    )


class LLMCodingAgent:
    """Isolated LLM agent responsible for generating and executing Python code based on instruction of LLMRecommendationAgent."""
    # Packages that LLM can use for cleaning data, besides built-in functions
    ALLOWED_PACKAGES = [
        'numpy',
        'pandas',
        'python-dateutil',
        'rapidfuzz',
        'fuzzywuzzy'
    ]
    def __init__(self, llm_client: BaseLLMClient,  llm_role: str, update_token_func, config: CleaningConfig):
        self.config = config
        self.system_prompt = "You are an expert data analyst specializing in tabular dataset cleaning. You MUST ONLY respond with an executable Python code block."
        self.llm_client = llm_client
        self.llm_role = llm_role
        self._update_token_func = update_token_func
        self.prompt_generator = PromptGeneration()

    async def _track_usage(self, token_usage):
        input_tokens = token_usage.get('input_tokens', 0)
        output_tokens = token_usage.get('output_tokens', 0)
        await self._update_token_func(input_tokens, output_tokens) 

    async def _call_llm_for_code_async(self, messages: list) -> tuple[str | None, str | None, str | None]:
        try:
            raw_response, token_usage = await self.llm_client.call_llm_async(
                messages,
                response_schema="text/plain",
                **llm_sampling_kwargs_from_config(self.config),
            )
            await self._track_usage(token_usage)
            if not raw_response:
                return None, None, "Empty response from LLM."
            
            code_block_pattern = "```(?:python|py)?\\s*\n?(.*?)\n?\\s*```"
            match = re.search(code_block_pattern, raw_response, re.DOTALL | re.IGNORECASE)
            if match:
                code_str = match.group(1).strip()
            else:
                code_str = raw_response.strip()

            if not code_str:
                return None, raw_response, "Extracted code string is empty."
            return code_str, raw_response, None
        except Exception as e:
            unavailable_msg = _service_unavailable_coder_message(e)
            if unavailable_msg:
                return None, None, unavailable_msg
            error_msg = f"Code Extraction Error: {type(e).__name__}: {e}"
            return None, None, error_msg
      
    @staticmethod
    async def _execute_code_async(code_str: str, df: pd.DataFrame, columns: str | list[str], timeout: int = 30) -> pd.Series | pd.DataFrame | str:
        """
        Runs _execute_code in a worker thread so the event loop keeps serving other tasks meanwhile.
        The input columns are copied first, on the event loop, because other tasks write into the same DataFrame.
        """
        input_df = df[[columns]].copy() if isinstance(columns, str) else df[list(columns)].copy()
        return await asyncio.to_thread(LLMCodingAgent._execute_code, code_str, input_df, columns, timeout)

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
        last_api_unavailable_msg: str | None = None
        for attempt in range(1, self.config.max_coding_attempts + 1):
            code_str, llm_output, llm_error_msg = await self._call_llm_for_code_async(retries_messages)
            if llm_output:
                retries_messages.append({"role": self.llm_role, "content": llm_output})
            if llm_error_msg:
                if llm_error_msg.startswith("API error (HTTP 503 Service Unavailable)"):
                    last_api_unavailable_msg = llm_error_msg
                    if attempt < self.config.max_coding_attempts:
                        await asyncio.sleep(3)
                    continue
                if llm_output:
                    fix_prompt = f"The previous attempt resulted in an empty code string. Please regenerate the entire valid code string now. The error was: {llm_error_msg}"
                else:             
                    fix_prompt = f"The previous attempt resulted in an API error. Please regenerate the entire valid code string now. The error was: {llm_error_msg}"
                retries_messages.append({"role": "user", "content": fix_prompt})
                continue
            # 3. Execute the LLM-generated code. If correct output type, return
            exec_result = await self._execute_code_async(code_str, df, col)
            if isinstance(exec_result, pd.Series):
                messages.append({"role": self.llm_role, "content": llm_output})
                return exec_result, code_str, messages
            else:
                exec_error_msg = exec_result
                fix_prompt = (
                    f"The Python code provided in the previous step failed during execution with the following error:"
                    f"\n\n{exec_error_msg}\n\n"
                    f"Please correct the code and provide the full, fixed executable code string again.")
                retries_messages.append({"role": "user", "content": fix_prompt})
        if last_api_unavailable_msg:
            return None, last_api_unavailable_msg, messages
        return None, None, messages
    
    async def clean_multi_col_async(self, 
                            df: pd.DataFrame, 
                            task_info: MultiColumnTask,
                            recommender_data: dict | None = None,
                            messages: list[dict[str, str]] | None = None
                            ) -> tuple[pd.DataFrame | None, str | None, list[dict] | None]:
        """Generates and executes cleaning code for multi-column operations. With retries for code-level errors."""
        target_cols = task_info.target_columns
        if messages is None:
            # 1. First time coder is called. Provide instructions
            messages = [{"role": "system", "content": self.system_prompt}]
            initial_prompt = self.prompt_generator.create_prompt_coding_multi_col(
                task_info, recommender_data, str(self.ALLOWED_PACKAGES))
            messages.append({"role": "user", "content": initial_prompt})
        # 2. Call LLM to generate code. With retries for incorrectly generated code
        retries_messages = messages.copy()
        last_api_unavailable_msg: str | None = None
        for attempt in range(1, self.config.max_coding_attempts + 1):
            code_str, llm_output, llm_error_msg = await self._call_llm_for_code_async(retries_messages)
            if llm_output:
                retries_messages.append({"role": self.llm_role, "content": llm_output})
            if llm_error_msg:
                if llm_error_msg.startswith("API error (HTTP 503 Service Unavailable)"):
                    last_api_unavailable_msg = llm_error_msg
                    if attempt < self.config.max_coding_attempts:
                        await asyncio.sleep(3)
                    continue
                if llm_output:
                    fix_prompt = f"The previous attempt resulted in an empty code string. Please regenerate the entire valid code string now. The error was: {llm_error_msg}"
                else:             
                    fix_prompt = f"The previous attempt resulted in an API error. Please regenerate the entire valid code string now. The error was: {llm_error_msg}"
                retries_messages.append({"role": "user", "content": fix_prompt})
                continue
            # 3. Execute the LLM-generated code. If result is correct output type, return it
            exec_result = await self._execute_code_async(code_str, df, target_cols)
            if isinstance(exec_result, pd.DataFrame):
                messages.append({"role": self.llm_role, "content": llm_output})
                return exec_result, code_str, messages
            else:
                exec_error_msg = exec_result
                fix_prompt = (
                    f"The Python code provided in the previous step failed during execution with the following error:"
                    f"\n\n{exec_error_msg}\n\n"
                    f"Please correct the code and provide the full, fixed executable code string again.")
                retries_messages.append({"role": "user", "content": fix_prompt})
        if last_api_unavailable_msg:
            return None, last_api_unavailable_msg, messages
        return None, None, messages