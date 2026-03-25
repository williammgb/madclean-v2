import pandas as pd
import asyncio
from functools import partial
# Local imports
from madclean.llm.llm_clients import BaseLLMClient
from madclean.components.multi_agent_cleaner.llm_coding import LLMCodingAgent
from madclean.components.multi_agent_cleaner.llm_validation import LLMValidationAgent
from madclean.components.multi_agent_cleaner.llm_recommending import LLMRecommendationAgent
from madclean.components.domain.schema import ColumnProfile, MultiColumnTask

class MultiAgentCleaning:
    """Multi-Agent LLM Cleaning Coordinator that coordinates the workflow between the LLM agents."""
    def __init__(self, llm_client: BaseLLMClient = None, llm_role: str = None, config: dict = None, agent_specs: dict | None = None):
        self.config = config
        self.verbose = self.config.verbose
        self.llm_client = llm_client
        self.llm_role = llm_role
        self.cancel_check = None
        self.token_usage = {
            "recommender": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "coding": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "validation": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        self._token_lock = asyncio.Lock()
        if agent_specs:
            recommender = agent_specs.get("recommender")
            coding = agent_specs.get("coding")
            validation = agent_specs.get("validation")
            if not recommender or not coding or not validation:
                raise ValueError("agent_specs must include recommender, coding, validation.")
            self.recommender_agent = LLMRecommendationAgent(recommender["client"], recommender["role"], partial(self._update_token_count, "recommender"), config=self.config)
            self.coding_agent = LLMCodingAgent(coding["client"], coding["role"], partial(self._update_token_count, "coding"), config=self.config)
            self.validation_agent = LLMValidationAgent(validation["client"], validation["role"], partial(self._update_token_count, "validation"), config=self.config)
        else:
            self.recommender_agent = LLMRecommendationAgent(llm_client, llm_role, partial(self._update_token_count, "recommender"), config=self.config)
            self.coding_agent = LLMCodingAgent(llm_client, llm_role, partial(self._update_token_count, "coding"), config=self.config)
            self.validation_agent = LLMValidationAgent(llm_client, llm_role, partial(self._update_token_count, "validation"), config=self.config)
        self.cleaning_report = {}

    def set_cancel_check(self, cancel_check):
        self.cancel_check = cancel_check

    def _check_cancelled(self):
        if callable(self.cancel_check) and self.cancel_check():
            raise asyncio.CancelledError()

    def reset_token_usage(self):
        self.token_usage = {
            "recommender": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "coding": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "validation": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }

    async def _update_token_count(self, agent, input_tokens, output_tokens):
        async with self._token_lock:
            usage = self.token_usage[agent]
            usage["input_tokens"] += input_tokens
            usage["output_tokens"] += output_tokens
            usage["total_tokens"] += input_tokens + output_tokens
    
    async def _run_column_cleaning_async(self, df: pd.DataFrame, col: str, column_profile: ColumnProfile) -> tuple[str, pd.Series | None, str]:
        """Manages the cleaning and verification workflow for a single column. This is the core interaction loop."""      
        column_type = column_profile.semantic_type
        recommender_history = None
        coder_history = None
        validator_history = None
        recommender_data = None
        cleaned_column = None 
        final_code_str = ""
        # 1. Run multi-agent cleaning loop. Start with Recommender Agent to generate instructions
        feedback_target = 'RECOMMENDER'
        for attempt in range(self.config.max_cleaning_attempts):
            self._check_cancelled()
            if feedback_target == 'RECOMMENDER':
                already_clean, recommender_data, new_recommender_history = await self.recommender_agent.generate_recommendations_async(
                    col, column_profile, messages=recommender_history)
                recommender_history = new_recommender_history 
                if already_clean:
                    msg = f"[{col}] Recommender Agent determined column is already clean."
                    self.cleaning_report[col] = {
                        "datatype": column_type,
                        "already_clean": True,
                        "cleaned": False,
                        "attempts": attempt + 1,
                        "generated_code": final_code_str,
                        "cleaning_validated": False
                    }
                    return col, df[col], msg
                if not recommender_data:
                    if attempt < self.config.max_cleaning_attempts - 1:
                        recommender_history.append({
                            "role": "user", 
                            "content": "Error: Your previous response was not a valid JSON or did not follow the schema. Please try again."
                        })
                        continue 
                    else:
                        break
                coder_history = None
            # 2. Pass instructions of Recommender Agent to Coding Agent
            cleaned_column, final_code_str, new_coder_history = await self.coding_agent.clean_column_async(
                df, col, column_type, 
                recommender_data=recommender_data, 
                messages=coder_history)
            coder_history = new_coder_history
            # 3. If Coding Agent could not generate valid code to clean column, send feedback to Recommender to provide better instructions
            if cleaned_column is None: 
                if attempt == self.config.max_cleaning_attemptss -1:
                    break
                feedback_target = 'RECOMMENDER'
                feedback_prompt = (
                    "Validation Feedback: Your last set of instructions caused the Coder agent to fail. "
                    "It produced invalid code. "
                    "This often means the instructions were too complex or ambiguous. "
                    "Please re-write your instructions."
                )
                if recommender_history:
                    recommender_history.append({"role": "user", "content": feedback_prompt})
                continue
            # 4. If column is cleaned, send to Validator Agent (if enabled)
            if not self.config.enable_validation:
                msg = f"[{col}] Successfully cleaned."
                self.cleaning_report[col] = {
                    "datatype": column_type,
                    "already_clean": False,
                    "cleaned": True,
                    "attempts": attempt + 1,
                    "generated_code": final_code_str,
                    "cleaning_validated": False
                }
                return col, cleaned_column, msg

            if column_type in ("INTEGER", "FLOAT", "BOOLEAN"):
                msg = f"[{col}] Successfully cleaned."
                self.cleaning_report[col] = {
                    "datatype": column_type,
                    "already_clean": False,
                    "cleaned": True,
                    "attempts": attempt + 1,
                    "generated_code": final_code_str,
                    "cleaning_validated": False,
                }
                return col, cleaned_column, msg

            last_attempt = attempt == (self.config.max_cleaning_attempts - 1)
            needs_correction, feedback_target, correction_instructions, new_validator_history = await self.validation_agent.validate_async(
                col, df[col], cleaned_column, column_type, messages=validator_history, last_attempt=last_attempt) 
            validator_history = new_validator_history
            # 5. If validator approves cleaned column, return column. Otherwise provide feedback to corresponding Agent
            if not needs_correction:
                msg = f"[{col}] Successfully cleaned and validated."
                self.cleaning_report[col] = {
                    "datatype": column_type,
                    "already_clean": False,
                    "cleaned": True,
                    "attempts": attempt + 1,
                    "generated_code": final_code_str,
                    "cleaning_validated": True
                }
                return col, cleaned_column, msg
            if attempt == self.config.max_cleaning_attempts - 1:
                break
            if self.verbose: print(f"[{col}] Validation Agent detected unintended changes or missed errors; updating cleaning operations.")
            if feedback_target == 'CODER':
                feedback_prompt = (
                    f"Validation Feedback: The previous code was almost correct, but it introduced undesired changes to the column format. Please fix it.\n\n"
                    f"Instructions: {correction_instructions}\n\n"
                    f"Provide the new valid, executable code string.")
                coder_history.append({"role": "user", "content": feedback_prompt})
            elif feedback_target == 'RECOMMENDER':
                feedback_prompt = (
                    f"Validation Feedback: Your last instructions were incomplete. The Coder agent followed them, but errors from the original data remain.\n\n"
                    f"Details: {correction_instructions}\n\n"
                    f"Please provide an updated and more complete set of cleaning instructions and examples based on this feedback.")
                recommender_history.append({"role": "user", "content": feedback_prompt})   
        msg = f"[{col}] FAILED cleaning after {self.config.max_cleaning_attemps} attempts."
        self.cleaning_report[col] = {
            "datatype": column_type,
            "already_clean": False,
            "cleaned": False,
            "attempts": self.config.max_cleaning_attemps,
            "generated_code": final_code_str,
             "cleaning_validated": False
        }
        return col, None, msg

    async def _run_multi_col_cleaning_async(self, df: pd.DataFrame, task_info: MultiColumnTask) -> tuple[list[str], pd.DataFrame | None, str]:
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
        for attempt in range(self.config.max_multi_col_attempts):
            self._check_cancelled()
            if feedback_target == 'RECOMMENDER':
                recommender_data, new_recommender_history = await self.recommender_agent.generate_recommendations_multi_col_async( 
                    df, task_info, messages=recommender_history)
                recommender_history = new_recommender_history 
                if not recommender_data: 
                    if attempt < self.config.max_multi_col_attempts - 1:
                        recommender_history.append({
                            "role": "user", 
                            "content": "Error: Your previous response was not a valid JSON or did not follow the schema. Please try again."
                        })
                        continue
                    else:
                        break
                coder_history = None
            # 2. Pass instructions of Recommender Agent to Coding Agent
            cleaned_targets, _final_code_str, new_coder_history = await self.coding_agent.clean_multi_col_async( 
                df, task_info, recommender_data, messages=coder_history)
            coder_history = new_coder_history
            # 3. If Coding Agent could not generate valid code, send feedback to Recommender to provide better instructions
            if cleaned_targets is None: 
                if attempt == self.config.max_multi_col_attempts -1:
                    break
                feedback_target = 'RECOMMENDER'
                feedback_prompt = (
                    "Validation Feedback: Your last set of instructions caused the Coding Agent to fail. "
                    "It produced invalid code. "
                    "This often means the instructions were too complex or ambiguous. "
                    "Please re-write your instructions."
                )
                if recommender_history:
                    recommender_history.append({"role": "user", "content": feedback_prompt})
                continue
            # 4. If columns are cleaned, send to Validator Agent 
            if not self.config.enable_validation_multi:
                msg = f"[{task_key}] Succesfully cleaned"
                self.cleaning_report[task_key] = {
                    "target_columns": target_cols,
                    "cleaned": True,
                    "attempts": attempt + 1,
                    "cleaning_validated": False
                }
                return target_cols, cleaned_targets, msg

            last_attempt = attempt == (self.config.max_multi_col_attempts - 1)
            needs_correction, feedback_target, correction_instructions, new_validator_history = await self.validation_agent.validate_multi_col_async(
                dirty_targets=df[target_cols],
                cleaned_targets=cleaned_targets,
                task_info=task_info,
                messages=validator_history,
                last_attempt=last_attempt)
            validator_history = new_validator_history
            # 5. If validator approves cleaned columns, return columns. Otherwise provide feedback to corresponding Agent
            if not needs_correction:
                msg = f"[{task_key}] Successfully cleaned and validated."
                self.cleaning_report[task_key] = {
                    "target_columns": target_cols,
                    "cleaned": True,
                    "attempts": attempt + 1,
                    "cleaning_validated": True
                }
                return target_cols, cleaned_targets, msg
            if attempt == self.config.max_multi_col_attempts - 1:
                break
            if feedback_target == 'CODER':
                feedback_prompt = (
                    f"Validation Feedback: The previous code was almost correct, but it introduced undesired changes to the column format. Please fix it.\n\n"
                    f"Instructions: {correction_instructions}\n\n"
                    f"Provide the new valid, executable code.")
                coder_history.append({"role": "user", "content": feedback_prompt})
            elif feedback_target == 'RECOMMENDER':
                feedback_prompt = (
                    f"Validation Feedback: Your last instructions were incomplete. The Coder agent followed them, but errors from the original data remain.\n\n"
                    f"Details: {correction_instructions}\n\n"
                    f"Please provide an updated and more complete set of cleaning instructions and examples based on this feedback.")
                recommender_history.append({"role": "user", "content": feedback_prompt})    
        msg = f"[{task_key}] FAILED cleaning after {self.config.max_multi_col_attempts} attempts."
        self.cleaning_report[task_key] = {
                    "target_columns": target_cols,
                    "cleaned": False,
                    "attempts": self.config.max_multi_col_attempts,
                    "cleaning_validated": False
                }
        return target_cols, None, msg


