import json
import re
import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator
from typing import Literal
# Local imports
from madclean.llm.llm_clients import BaseLLMClient
from madclean.components.coordinator.prompt_generation import PromptGeneration
from madclean.components.domain.schema import MultiColumnTask
from madclean.config.settings import CleaningConfig
from madclean.utils.helpers import llm_sampling_kwargs_from_config, seeded_random

MAX_VALIDATION_CASES = 20


class ValidationCase(BaseModel):
    """One cell the validator objects to, as the overview showed it."""
    original: str | None = None
    cleaned: str | None = None
    expected: str | None = None
    problem: str = ""

    @field_validator("original", "cleaned", "expected", "problem", mode="before")
    @classmethod
    def _as_text(cls, value):
        # Models write numeric cells as JSON numbers; the overview showed them as text.
        if value is None or isinstance(value, str):
            return value
        return str(value)


class CodeOutputValidation(BaseModel):
    """Defines the required JSON output strcuture for verification of Gemini API."""
    # analysis comes first so the model reasons before it decides. The fields that are new in v3
    # have defaults, so answers without them (older models, scripted fakes) still parse.
    analysis: str = ""
    needs_correction: bool
    issue_kind: Literal["OVER_CLEANING", "MISSED_ERRORS", "FORMAT_CHANGE", "CODE_BUG"] | None = None
    feedback_target: Literal["CODER", "RECOMMENDER"] | None
    correction_instructions: str
    cases: list[ValidationCase] | None = None


def validation_details(parsed_response: dict | None) -> dict:
    """The verdict's kind and cases, which the cleaning loop words its feedback with."""
    parsed_response = parsed_response or {}
    cases = parsed_response.get("cases")
    return {
        "issue_kind": parsed_response.get("issue_kind"),
        "cases": list(cases)[:MAX_VALIDATION_CASES] if cases else None,
    }

class LLMValidationAgent:
    """Isolated LLM agent that validates the performed cleaning operations, gives feedback if operations are incorrect."""
    def __init__(self, llm_client: BaseLLMClient, llm_role: str, update_token_func, config: CleaningConfig):
        self.config = config
        self.llm_client = llm_client
        self.llm_role = llm_role
        self.system_prompt = "You are a data validation expert tasked with evaluating if a cleaned dataset remains consistent with the original data, detecting unwanted or excessive transformations."
        self._update_token_func = update_token_func
        self.prompt_generator = PromptGeneration()
    
    async def _track_usage(self, token_usage):
        input_tokens = token_usage.get('input_tokens', 0)
        output_tokens = token_usage.get('output_tokens', 0)
        await self._update_token_func(input_tokens, output_tokens)

    async def _call_llm_with_parsing(self, messages: list, schema: type[BaseModel]) -> tuple[dict, str]:
        working_messages = messages.copy()
        required_keys = list(schema.model_fields.keys())
        for attempt in range(self.config.max_parse_attempts):
            raw_response, token_usage = await self.llm_client.call_llm_async(
                working_messages,
                response_schema=schema,
                **llm_sampling_kwargs_from_config(self.config),
            )
            await self._track_usage(token_usage)
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
                if validated_data.needs_correction:
                    if not validated_data.correction_instructions or not validated_data.correction_instructions.strip():
                        raise ValueError("You indicated 'needs_correction': true, but 'correction_instructions' are empty.")
                    if not validated_data.feedback_target:
                        raise ValueError("You indicated 'needs_correction': true, but 'feedback_target' is missing.")

                return validated_data.model_dump(), raw_response
            except (json.JSONDecodeError, ValueError, ValidationError, KeyError) as e:
                if attempt == self.config.max_parse_attempts - 1:
                    raise RuntimeError(f"Validation parsing failed after {self.config.max_parse_attempts} attempts. Error: {e}")
                working_messages.append({"role": self.llm_role, "content": raw_response})
                fix_prompt = (
                    f"The JSON provided is invalid or semantically incorrect. Error: {str(e)}. "
                    f"Please provide the full valid JSON object with keys: {required_keys}. "
                    "Ensure 'correction_instructions' is filled if 'needs_correction' is true."
                )
                working_messages.append({"role": "user", "content": fix_prompt})

    async def validate_async(self, 
               col: str, 
               dirty_column: pd.Series, 
               cleaned_column: pd.Series,
               column_type: str, 
               messages: list[dict[str, str]] | None = None,
               last_attempt: bool = False,
               attempt: int = 0) -> tuple[bool, str | None, str| None, list[dict[str, str]], str, dict]:
        """
        Compares dirty and cleaned columns and validates cleaning operations.
        If validator detects undesired changes, it sends feedback to the corresponding LLM agent.
        Returns (needs_correction, feedback_target, correction_instructions, messages, raw_llm_output, details),
        where details is {"issue_kind": ..., "cases": [...]}. The agent is shared by columns cleaned
        concurrently, so the details travel with the answer instead of living on the agent.
        """
        # 1. Using validator for INTEGER, FLOAT and BOOLEAN types is unnecessary
        if column_type in ("INTEGER", "FLOAT", "BOOLEAN"):
            return False, None, None, messages, "", validation_details(None)
        # 2. Create validation prompt
        if messages is None:
            messages = [{"role": "system", "content": self.system_prompt}]
        prompt = self.prompt_generator.create_prompt_validation(
            col,
            dirty_column,
            cleaned_column,
            column_type,
            last_attempt,
            max_sample_size=self.config.sample_size_validator,
            random_sample_size=self.config.sample_size_validator_random,
            changed_sample_size=self.config.sample_size_validator_changed,
            rng=seeded_random(self.config.sampling_seed, "validation", col, attempt),
        )
        hints = getattr(self.config, "recommender_column_hints", None) or {}
        if isinstance(hints, dict):
            c_hint = str((hints.get(col) or "")).strip()
            if c_hint:
                prompt += (
                    "\n\n### USER-DEFINED CLEANING CONSTRAINTS\n"
                    "Apply these user constraints when judging whether a change is acceptable:\n"
                    f"{c_hint}\n"
                )
        labeled_map = getattr(self.config, "recommender_column_labeled_examples", None) or {}
        if isinstance(labeled_map, dict):
            c_lab = str((labeled_map.get(col) or "")).strip()
            if c_lab:
                prompt += (
                    "\n\n### USER-LABELED CELL EXAMPLES (authoritative for those rows)\n"
                    "The user marked specific cells in this column. Treat these as ground truth for what is "
                    "acceptable (CLEAN) or what the cleaned value must become (DIRTY → expected). "
                    "Do not flag corrections that move a cell toward these targets. "
                    "If the cleaned column contradicts a DIRTY expectation or alters a cell marked CLEAN without fixing an error, that is an undesired change.\n"
                    f"{c_lab}\n"
                )
        messages.append({"role": "user", "content": prompt})
        # 3. Call Validation Agent to validate cleaning operations
        try:
            parsed_response, raw_response = await self._call_llm_with_parsing(messages, schema=CodeOutputValidation)
            messages.append({"role": self.llm_role, "content": raw_response})

            # 4. If validation failed, send feedback. Otherwise accept cleaned column
            if parsed_response["needs_correction"]:
                return (True, parsed_response["feedback_target"] , parsed_response["correction_instructions"], messages,
                        raw_response, validation_details(parsed_response))
            return False, None, None, messages, raw_response, validation_details(None)
        except Exception:
            return self._failure_verdict(messages)

    def _failure_verdict(self, messages) -> tuple:
        """What a validator that could not answer returns, by validator_failure_strategy."""
        strategy = getattr(self.config, "validator_failure_strategy", "accept_cleaned")
        if strategy == "leave_uncleaned":
            return True, "RECOMMENDER", "__VALIDATOR_FAILURE_LEAVE_UNCLEANED__", messages, "", validation_details(None)
        if strategy == "ask_user":
            return True, "RECOMMENDER", "__VALIDATOR_FAILURE_ASK_USER__", messages, "", validation_details(None)
        return False, None, None, messages, "", validation_details(None)

    async def validate_multi_col_async(self,
                dirty_targets: pd.DataFrame,
                cleaned_targets: pd.DataFrame,
                task_info: MultiColumnTask,
                messages: list[dict[str, str]] | None = None,
                last_attempt: bool = False,
                attempt: int = 0) -> tuple[bool, str | None, str | None, list[dict[str, str]], str, dict]:
        """
        Compares dirty and cleaned column pairs and validates cleaning operations.
        If validator detects undesired changes, it sends feedback to the corresponding LLM agent.
        Returns the same six elements as validate_async.
        """
        # 1. Create validation prompt
        if messages is None:
            messages = [{"role": "system", "content": self.system_prompt}]
        prompt = self.prompt_generator.create_prompt_validation_multi_col(
            dirty_targets,
            cleaned_targets,
            task_info,
            last_attempt,
            max_sample_size=self.config.sample_size_validator,
            random_sample_size=self.config.sample_size_validator_random,
            changed_sample_size=self.config.sample_size_validator_changed,
            rng=seeded_random(self.config.sampling_seed, "validation", task_info.verbose_key, attempt),
        )
        messages.append({"role": "user", "content": prompt})
        # 2. Call Validation Agent
        try:
            parsed_response, raw_response = await self._call_llm_with_parsing(messages, schema=CodeOutputValidation)
            messages.append({"role": self.llm_role, "content": raw_response})
            if parsed_response["needs_correction"]:
                return (True, parsed_response["feedback_target"], parsed_response["correction_instructions"], messages,
                        raw_response, validation_details(parsed_response))
            return False, None, None, messages, raw_response, validation_details(None)
        except Exception:
            return self._failure_verdict(messages)