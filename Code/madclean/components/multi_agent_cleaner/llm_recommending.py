import json
import re
from pydantic import BaseModel, ValidationError
import pandas as pd
# Local imports
from madclean.llm.llm_clients import BaseLLMClient
from madclean.components.coordinator.prompt_generation import PromptGeneration
from madclean.components.domain.schema import ColumnProfile, MultiColumnTask
from madclean.config.settings import CleaningConfig
from madclean.utils.helpers import llm_sampling_kwargs_from_config


def _service_unavailable_recommender_payload(exc: BaseException) -> dict | None:
    """Map HTTP 503 (and similar upstream overload) to a trace-friendly recommender payload."""
    code = getattr(exc, "status_code", None)
    if code is None:
        resp = getattr(exc, "response", None)
        if resp is not None:
            code = getattr(resp, "status_code", None)
    if code != 503:
        return None
    return {
        "_api_error": True,
        "http_status": 503,
        "summary": (
            "API error (HTTP 503 Service Unavailable): the LLM provider is temporarily unavailable. "
            "This is not a failure of the cleaning system. System will retry in a few seconds."
        ),
    }


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

class LLMRecommendationAgent:
    """Isolated LLM Agent that analyses data provided by DataProfiler and generates cleaning instructions for LLMCodingAgent."""
    
    MAX_PARSE_ATTEMPTS = 3

    def __init__(self, llm_client: BaseLLMClient,  llm_role: str, update_token_func, config: CleaningConfig):
        self.config = config
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
    async def _track_usage(self, token_usage):
        input_tokens = token_usage.get('input_tokens', 0)
        output_tokens = token_usage.get('output_tokens', 0)
        await self._update_token_func(input_tokens, output_tokens)

    # ====== Updates for robustness ==========
    async def _call_llm_with_parsing(self, messages: list, schema: type[BaseModel]) -> tuple[dict, str]:
        """Helper function to call LLM, parse JSON and retry on failure. """
        working_messages = messages.copy()
        # 1. Extract required keys from schema
        required_keys = list(schema.model_fields.keys())
        # 2. Call LLM and parse output, with retries
        for attempt in range(self.MAX_PARSE_ATTEMPTS):
            try:
                raw_response, token_usage = await self.llm_client.call_llm_async(
                    working_messages,
                    response_schema=schema,
                    **llm_sampling_kwargs_from_config(self.config),
                )
                await self._track_usage(token_usage)
            except Exception as call_exc:
                api_payload = _service_unavailable_recommender_payload(call_exc)
                if api_payload is not None and attempt < self.MAX_PARSE_ATTEMPTS - 1:
                    continue
                raise

            try:
                # Remove noise and extract JSON
                clean_raw = raw_response.strip().replace("```json", "").replace("```", "")
                json_match = re.search(r'\{.*\}', clean_raw, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON object found in response.")
                json_string = json_match.group(0)
                parsed_data = json.loads(json_string)
                # Validate against schema
                validated_data = schema(**parsed_data)

                # Check that instructions are actually present
                is_clean_val = getattr(validated_data, 'is_clean', False)
                # single columns
                if hasattr(validated_data, 'cleaning_instructions'):
                    instr = validated_data.cleaning_instructions
                    if not is_clean_val and (not instr or all(not s.strip() for s in instr)):
                        raise ValueError("Column marked as 'dirty' but instructions are missing or empty.")
                # FDs
                if hasattr(validated_data, 'violation_instructions'):
                    if not validated_data.violation_instructions.strip() and not validated_data.imputation_instructions.strip():
                        raise ValueError("FD task requires explicit instructions.")
                return validated_data.model_dump(), raw_response
            
            except (json.JSONDecodeError, ValueError, ValidationError, KeyError) as e:
                if attempt == self.MAX_PARSE_ATTEMPTS - 1:
                    raise RuntimeError(f"Failed to get valid JSON after {self.MAX_PARSE_ATTEMPTS} tries. Last error: {e}")
                
                working_messages.append({"role": self.llm_role, "content": raw_response})
                fix_prompt = (
                    f"Your previous response was invalid. Error: {str(e)}. "
                    f"Please output a valid JSON object strictly following this schema keys: {required_keys}. "
                    "Do not include any conversational text."
                )
                working_messages.append({"role": "user", "content": fix_prompt})

    async def generate_recommendations_async(self, col: str, column_profile: ColumnProfile, 
                                            messages: list | None = None) -> tuple[bool, dict | None, list | None]:
        if messages is None:
            # 1. First time the recommender agent is called. Provide profiler data.
            messages = [{"role": "system", "content": self.system_prompt}]
            hints = getattr(self.config, "recommender_column_hints", None) or {}
            extra = (hints.get(col) or "").strip() if isinstance(hints, dict) else ""
            if not extra:
                extra = (getattr(self.config, "recommender_extra_instructions", None) or "").strip()
            labeled_map = getattr(self.config, "recommender_column_labeled_examples", None) or {}
            labeled = (labeled_map.get(col) or "").strip() if isinstance(labeled_map, dict) else ""
            initial_prompt = self.prompt_generator.create_prompt_recommender(
                col,
                column_profile,
                user_constraints=extra,
                labeled_examples=labeled,
            )
            messages.append({"role": "user", "content": initial_prompt})
        
        try:
            parsed_response, raw_response = await self._call_llm_with_parsing(messages, schema=CodeOutputRecommendation)
            is_clean = parsed_response.get('is_clean', False)
            messages.append({"role": self.llm_role, "content": raw_response}) 
            if is_clean:
                return True, None, None
            return False, parsed_response, messages
        except Exception as e:
            api_payload = _service_unavailable_recommender_payload(e)
            if api_payload is not None:
                return False, api_payload, messages
            return False, None, messages

    
    async def generate_recommendations_multi_col_async(self, df: pd.DataFrame, task_info: MultiColumnTask, 
                                                       messages: list | None = None) -> tuple[dict | None, list | None]:
        config = self.multi_col_config.get(task_info.task_type)
        if not config:
            raise ValueError(f"Unsupported task: {task_info.task_type}")
            
        if messages is None:
            # 1. First time the recommender agent is called. Provide multi-column data.
            messages = [{"role": "system", "content": config['system_prompt']}]
            hints = getattr(self.config, "recommender_column_hints", None) or {}
            labeled_map = getattr(self.config, "recommender_column_labeled_examples", None) or {}
            extra_parts: list[str] = []
            labeled_parts: list[str] = []
            if isinstance(hints, dict):
                for c in task_info.target_columns or []:
                    cs = str(c)
                    h = (hints.get(cs) or "").strip()
                    if h:
                        extra_parts.append(f"- {cs}: {h}")
            extra = "\n".join(extra_parts).strip()
            if not extra:
                extra = (getattr(self.config, "recommender_extra_instructions", None) or "").strip()
            if isinstance(labeled_map, dict):
                for c in task_info.target_columns or []:
                    cs = str(c)
                    block = (labeled_map.get(cs) or "").strip()
                    if block:
                        labeled_parts.append(f"### Column {cs}\n{block}")
            labeled = "\n\n".join(labeled_parts).strip()
            initial_prompt = self.prompt_generator.create_prompt_recommender_multi_col(
                df,
                task_info,
                user_constraints=extra,
                labeled_examples=labeled,
            )
            messages.append({"role": "user", "content": initial_prompt})
        
        try:
            parsed_response, raw_response = await self._call_llm_with_parsing(messages, schema=config['schema'])
            messages.append({"role": self.llm_role, "content": raw_response})
            return parsed_response, messages
        except Exception as e:
            api_payload = _service_unavailable_recommender_payload(e)
            if api_payload is not None:
                return api_payload, messages
            return None, messages