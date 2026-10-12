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
from madclean.config.settings import CleaningConfig
from madclean.utils.helpers import llm_sampling_kwargs_from_config
from madclean.components.multi_agent_cleaner.code_checks import (
    emptied_feedback,
    emptied_inputs,
    emptied_share,
    example_cases,
    example_failures,
    example_feedback,
    example_inputs,
    example_note,
    wrap_with_value_map,
)

GENERATED_CODE_FILE = "generated_code.py"
TRACEBACK_LINES = 15

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
            # The code gets a file of its own, so a traceback names its own line numbers ("line 3").
            code_path = os.path.join(tmpdir, GENERATED_CODE_FILE)
            input_file = os.path.join(tmpdir, 'input.pkl')
            output_file = os.path.join(tmpdir, 'result.pkl')
            input_dict = {
                'data': input_data,
                'is_series_input': is_series_input
            }
            with open(input_file, 'wb') as f:
                pickle.dump(input_dict, f)
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code_str)
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

with open("{GENERATED_CODE_FILE}", encoding="utf-8") as f:
    exec(compile(f.read(), "{GENERATED_CODE_FILE}", "exec"))

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
                    text=True,
                    cwd=tmpdir,
                )
                result_file = os.path.join(tmpdir, "result.pkl")
                with open(result_file, "rb") as f:
                    return pickle.load(f)
            except subprocess.TimeoutExpired:
                return f"Execution did not finish before {timeout} seconds timeout. Check for blocking statements."
            except subprocess.CalledProcessError as e:
                # The tail of the traceback: the failing line of the code and the value that broke it.
                stderr_lines = e.stderr.strip().splitlines()
                return "\n".join(stderr_lines[-TRACEBACK_LINES:])

    async def clean_column_async(self, 
                            df: pd.DataFrame, 
                            col: str, 
                            column_type: str,
                            recommender_data: dict | None = None,
                            messages: list[dict[str, str]] | None = None
                            ) -> tuple[pd.Series | None, str | None, list[dict] | None, str | None]:
        """Generates and executes cleaning code for single column. With retries for code-level errors.

        Each attempt: wrap the code with the value table, run it, reject it if it empties filled cells,
        then check it against the Recommender's examples. Returns the column, the final code, the coder
        history and a note for the trace (the example check), or None for the column when nothing passed.
        A value table without instructions needs no Coder: the code is the table alone and history is None.
        """
        recommender_data = recommender_data or {}
        value_mapping = recommender_data.get("value_mapping")
        instructions = [s for s in recommender_data.get("cleaning_instructions") or [] if str(s).strip()]
        if value_mapping and not instructions and messages is None:
            table_code = wrap_with_value_map(None, value_mapping)
            exec_result = await self._execute_code_async(table_code, df, col)
            if isinstance(exec_result, pd.Series):
                return exec_result, table_code, None, "Value table only: the Coder was not called."
            return None, f"The value table's code failed: {exec_result}", None, None
        cases = example_cases(recommender_data)
        empty_inputs = emptied_inputs(recommender_data)
        best = None  # (failed examples, column, code, coder answer) of the closest attempt
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
            # 3. Wrap the code with the value table and run it. Wrong output type or a crash goes back.
            final_code = wrap_with_value_map(code_str, value_mapping)
            exec_result = await self._execute_code_async(final_code, df, col)
            if not isinstance(exec_result, pd.Series):
                fix_prompt = (
                    f"The Python code provided in the previous step failed during execution with the following error:"
                    f"\n\n{exec_result}\n\n"
                    f"Please correct the code and provide the full, fixed executable code string again.")
                retries_messages.append({"role": "user", "content": fix_prompt})
                continue
            # 4. Reject code that empties most filled cells that were not placeholders.
            emptied, filled = emptied_share(df[col], exec_result, empty_inputs)
            if filled and emptied / filled > self.config.max_emptied_share:
                fix_prompt = (
                    f"{emptied_feedback(emptied, filled)}\n\n"
                    "Please correct the code and provide the full, fixed executable code string again.")
                retries_messages.append({"role": "user", "content": fix_prompt})
                continue
            # 5. The Recommender's examples must come out as it said.
            note = None
            if cases:
                inputs = pd.DataFrame({col: example_inputs(df[col], [source for source, _ in cases])})
                failures = example_failures(await self._execute_code_async(final_code, inputs, col), cases)
                note = example_note(len(failures), len(cases))
                if failures:
                    if best is None or len(failures) < best[0]:
                        best = (len(failures), exec_result, final_code, llm_output)
                    retries_messages.append({"role": "user", "content": example_feedback(failures, len(cases))})
                    continue
            messages.append({"role": self.llm_role, "content": llm_output})
            return exec_result, final_code, messages, note
        # 6. No attempt passed every example: the closest one that ran and passed the guard goes on.
        if best is not None:
            failed, exec_result, final_code, llm_output = best
            messages.append({"role": self.llm_role, "content": llm_output})
            return exec_result, final_code, messages, example_note(failed, len(cases))
        if last_api_unavailable_msg:
            return None, last_api_unavailable_msg, messages, None
        return None, None, messages, None