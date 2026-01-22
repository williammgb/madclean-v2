import pandas as pd
import asyncio
# Local imports
from macs.llm.llm_clients import BaseLLMClient
from macs.components.multi_agent_cleaner.llm_coding import LLMCodingAgent
from macs.components.multi_agent_cleaner.llm_validation import LLMValidationAgent
from macs.components.multi_agent_cleaner.llm_recommending import LLMRecommendationAgent
from macs.components.domain.schema import ColumnProfile, MultiColumnTask

class MultiAgentCleaning:
    MAX_CLEANING_ATTEMPTS = 5
    MAX_MULTI_COL_ATTEMPTS = 3
    
    def __init__(self, llm_client: BaseLLMClient,  llm_role: str, verbose: bool = False):
        """Multi-Agent LLM Cleaning Coordinator that coordinates the workflow between the LLM agents."""
        self.verbose = verbose
        self.llm_client = llm_client
        self.llm_role = llm_role     
        self.token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0 }
        self._token_lock = asyncio.Lock()
        self.recommender_agent = LLMRecommendationAgent(llm_client, llm_role, self._update_token_count, verbose=self.verbose)
        self.coding_agent = LLMCodingAgent(llm_client, llm_role, self._update_token_count, verbose=self.verbose)
        self.validation_agent = LLMValidationAgent(llm_client, llm_role, self._update_token_count, verbose=self.verbose)
    
    def reset_token_usage(self):
        """Resets token usage counter for each new file."""
        self.token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0 }

    async def _update_token_count(self, input_tokens: int, output_tokens: int):
        """Method to update token usage, thread-safe due to use of asynchronous functions."""
        async with self._token_lock:
            self.token_usage['input_tokens'] += input_tokens
            self.token_usage['output_tokens'] += output_tokens
            self.token_usage['total_tokens'] += (input_tokens + output_tokens)
    
    async def _run_column_cleaning_async(self, df: pd.DataFrame, col: str, column_profile: ColumnProfile) -> tuple[str, pd.Series | None]:
        """Manages the cleaning and verification workflow for a single column. This is the core interaction loop."""      
        column_type = column_profile.semantic_type
        recommender_history = None
        coder_history = None
        validator_history = None
        recommender_data = None
        cleaned_column = None 
        # 1. Run multi-agent cleaning loop. Start with Recommender Agent to generate instructions
        feedback_target = 'RECOMMENDER'
        for attempt in range(self.MAX_CLEANING_ATTEMPTS):
            # if self.verbose: print(f"[{col}] Cleaning attempt {attempt + 1}/{self.MAX_CLEANING_ATTEMPTS}...")  
            if feedback_target == 'RECOMMENDER':
                # if self.verbose and attempt > 0: print(f"[{col}] Updating recommendations...")
                already_clean, recommender_data, new_recommender_history = await self.recommender_agent.generate_recommendations_async(
                    col, column_profile, messages=recommender_history)
                recommender_history = new_recommender_history 
                if already_clean:
                    if self.verbose: print(f"[{col}] Already clean.")
                    return col, df[col]
                if not recommender_data:
                    return col, None
                coder_history = None
            # 2. Pass instructions of Recommender Agent to Coding Agent
            cleaned_column, new_coder_history = await self.coding_agent.clean_column_async(
                df, col, column_type, 
                recommender_data=recommender_data, 
                messages=coder_history)
            coder_history = new_coder_history
            # 3. If Coding Agent could not generate valid code to clean column, send feedback to Recommender to provide better instructions
            if cleaned_column is None: 
                if attempt == self.MAX_CLEANING_ATTEMPTS -1:
                    break
                feedback_target = 'RECOMMENDER'
                feedback_prompt = (
                    "Validation Feedback: Your last set of instructions caused the Coder agent to fail. "
                    "It produced invalid code or a non-JSON output. "
                    "This often means the instructions were too complex or ambiguous. "
                    "Please re-write your instructions."
                )
                if recommender_history:
                    recommender_history.append({"role": "user", "content": feedback_prompt})
                continue
            # 4. If column is cleaned, send to Validator Agent
            last_attempt = attempt == (self.MAX_CLEANING_ATTEMPTS - 1)
            needs_correction, feedback_target, correction_instructions, new_validator_history = await self.validation_agent.validate_async(
                col, df[col], cleaned_column, column_type, messages=validator_history, last_attempt=last_attempt) 
            validator_history = new_validator_history
            # 5. If validator approves cleaned column, return column. Otherwise provide feedback to corresponding Agent
            if not needs_correction:
                if self.verbose: print(f"[{col}] Successfully cleaned and validated.")
                return col, cleaned_column
            if attempt == self.MAX_CLEANING_ATTEMPTS - 1:
                break
            if feedback_target == 'CODER':
                feedback_prompt = (
                    f"Validation Feedback: The previous code was almost correct, but it introduced undesired changes to the column format. Please fix it.\n\n"
                    f"Instructions: {correction_instructions}\n\n"
                    f"Provide the new JSON object with the corrected 'code'.")
                coder_history.append({"role": "user", "content": feedback_prompt})
            elif feedback_target == 'RECOMMENDER':
                feedback_prompt = (
                    f"Validation Feedback: Your last instructions were incomplete. The Coder agent followed them, but errors from the original data remain.\n\n"
                    f"Details: {correction_instructions}\n\n"
                    f"Please provide an updated and more complete set of cleaning instructions and examples based on this feedback.")
                recommender_history.append({"role": "user", "content": feedback_prompt})   
        if self.verbose: print(f"[{col}] FAILED cleaning after {self.MAX_CLEANING_ATTEMPTS} attempts.")
        return col, None

    async def _run_multi_col_cleaning_async(self, df: pd.DataFrame, task_info: MultiColumnTask) -> tuple[list[str], pd.DataFrame | None]:
        """Manages the cleaning and verification workflow for a multi-column operations. This is the core interaction loop."""          
        recommender_history = None
        coder_history = None
        validator_history = None
        recommender_data = None
        cleaned_targets = None 
        target_cols = task_info.target_columns
        task_key = task_info.verbose_key
        # 1. Run multi-agent cleaning loop. Start with RecommenderAgent
        feedback_target = 'RECOMMENDER'
        for attempt in range(self.MAX_MULTI_COL_ATTEMPTS):
            if feedback_target == 'RECOMMENDER':
                # if self.verbose: print(f"[{task_key}] Cleaning attempt {attempt + 1}/{self.MAX_MULTI_COL_ATTEMPTS}...") 
                recommender_data, new_recommender_history = await self.recommender_agent.generate_recommendations_multi_col_async( 
                    df, task_info, messages=recommender_history)
                recommender_history = new_recommender_history 
                if not recommender_data:
                    return target_cols, None
                coder_history = None
            # 2. Pass instructions of Recommender Agent to Coding Agent
            cleaned_targets, new_coder_history = await self.coding_agent.clean_multi_col_async( 
                df, task_info, recommender_data, messages=coder_history)
            coder_history = new_coder_history
            # 3. If Coding Agent could not generate valid code, send feedback to Recommender to provide better instructions
            if cleaned_targets is None: 
                if attempt == self.MAX_MULTI_COL_ATTEMPTS -1:
                    break
                feedback_target = 'RECOMMENDER'
                feedback_prompt = (
                    "Validation Feedback: Your last set of instructions caused the Coder agent to fail. "
                    "It produced invalid code or a non-JSON output. "
                    "This often means the instructions were too complex or ambiguous. "
                    "Please re-write your instructions."
                )
                if recommender_history:
                    recommender_history.append({"role": "user", "content": feedback_prompt})
                continue
            # 4. If columns are cleaned, send to Validator Agent
            last_attempt = attempt == (self.MAX_MULTI_COL_ATTEMPTS - 1)
            needs_correction, feedback_target, correction_instructions, new_validator_history = await self.validation_agent.validate_multi_col_async(
                dirty_targets=df[target_cols],
                cleaned_targets=cleaned_targets,
                task_info=task_info,
                messages=validator_history,
                last_attempt=last_attempt)
            validator_history = new_validator_history
            # 5. If validator approves cleaned columns, return columns. Otherwise provide feedback to corresponding Agent
            if not needs_correction:
                if self.verbose: print(f"[{task_key}] Successfully cleaned and validated.")
                return target_cols, cleaned_targets
            if attempt == self.MAX_MULTI_COL_ATTEMPTS - 1:
                break
            if feedback_target == 'CODER':
                feedback_prompt = (
                    f"Validation Feedback: The previous code was almost correct, but it introduced undesired changes to the column format. Please fix it.\n\n"
                    f"Instructions: {correction_instructions}\n\n"
                    f"Provide the new JSON object with the corrected 'code'.")
                coder_history.append({"role": "user", "content": feedback_prompt})
            elif feedback_target == 'RECOMMENDER':
                feedback_prompt = (
                    f"Validation Feedback: Your last instructions were incomplete. The Coder agent followed them, but errors from the original data remain.\n\n"
                    f"Details: {correction_instructions}\n\n"
                    f"Please provide an updated and more complete set of cleaning instructions and examples based on this feedback.")
                recommender_history.append({"role": "user", "content": feedback_prompt})    
        if self.verbose: print(f"[{task_key}] FAILED cleaning after {self.MAX_MULTI_COL_ATTEMPTS} attempts.")
        return target_cols, None


