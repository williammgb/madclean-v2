import pandas as pd
import asyncio
import json
import random
import time
from contextvars import ContextVar
from functools import partial
# Local imports
from madclean.llm.llm_clients import BaseLLMClient
from madclean.components.multi_agent_cleaner.llm_coding import LLMCodingAgent
from madclean.components.multi_agent_cleaner.llm_validation import LLMValidationAgent
from madclean.components.multi_agent_cleaner.llm_recommending import LLMRecommendationAgent
from madclean.components.domain.schema import ColumnProfile, MultiColumnTask
from madclean.components.domain.report import (
    AgentTokenUsage,
    ColumnReport,
    FDReport,
    TokenUsage,
    TraceEvent,
    TraceStep,
)
from madclean.utils.helpers import align_dirty_cleaned_series, seeded_random

# Which column or dependency task the calls of this asyncio task belong to. Columns are cleaned
# concurrently through one shared set of agents, so the task's own context is the only place that
# knows whose tokens are being counted. Each asyncio task gets its own copy.
_current_task_key: ContextVar[str | None] = ContextVar("madclean_task_key", default=None)

class MultiAgentCleaning:
    """Multi-Agent LLM Cleaning Coordinator that coordinates the workflow between the LLM agents."""
    def __init__(
        self,
        llm_client: BaseLLMClient = None,
        llm_role: str = None,
        config: dict = None,
        agent_specs: dict | None = None,
        trace_callback=None,
        user_validation_callback=None,
        hitl_callback=None,
    ):
        self.config = config
        self.verbose = self.config.verbose
        self.llm_client = llm_client
        self.llm_role = llm_role
        self.cancel_check = None
        self.trace_callback = trace_callback
        self.user_validation_callback = user_validation_callback
        self.hitl_callback = hitl_callback
        self.token_usage = AgentTokenUsage()
        # Tokens per column or dependency task, for the agent statistics. It is kept beside the
        # report rather than inside it, so the report's dictionary form does not change shape.
        self.per_task_usage: dict[str, TokenUsage] = {}
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
        self.cleaning_report: dict[str, ColumnReport | FDReport] = {}

    def set_cancel_check(self, cancel_check):
        self.cancel_check = cancel_check

    def _check_cancelled(self):
        if callable(self.cancel_check) and self.cancel_check():
            raise asyncio.CancelledError()

    def reset_token_usage(self):
        self.token_usage = AgentTokenUsage()
        self.per_task_usage = {}

    @staticmethod
    def _trace_text(payload) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        try:
            return json.dumps(payload, indent=2, default=str)
        except Exception:
            return str(payload)

    @staticmethod
    def _recommender_trace_status(recommender_data, already_clean: bool = False) -> str:
        if already_clean:
            return "already_clean"
        if isinstance(recommender_data, dict) and recommender_data.get("_api_error"):
            return "api_unavailable"
        if recommender_data:
            return "ok"
        return "invalid_response"

    @staticmethod
    def _format_recommender_output(recommender_data, already_clean: bool = False) -> str:
        if already_clean:
            return "Recommender Agent has determined column is already clean."
        if isinstance(recommender_data, dict) and recommender_data.get("_api_error"):
            return str(recommender_data.get("summary", "") or "").strip() or (
                "API error (HTTP 503): LLM provider unavailable."
            )
        if not isinstance(recommender_data, dict):
            return ""
        parts = []
        summary = str(recommender_data.get("summary", "") or "").strip()
        if summary:
            parts.append(f"Summary:\n{summary}")
        for label, key in [
            ("Error Types", "error_types"),
            ("Cleaning Instructions", "cleaning_instructions"),
            ("Dirty Examples", "examples_dirty"),
            ("Clean Examples", "examples_clean"),
        ]:
            val = recommender_data.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, list):
                rendered_rows = []
                for v in val:
                    if isinstance(v, (dict, list)):
                        rendered_rows.append(f"- {json.dumps(v, default=str)}")
                    else:
                        rendered_rows.append(f"- {str(v)}")
                rendered = "\n".join(rendered_rows)
            else:
                rendered = str(val)
            parts.append(f"{label}:\n{rendered}")
        return "\n\n".join(parts).strip()

    @staticmethod
    def _format_validator_output(needs_correction, feedback_target, correction_instructions) -> str:
        if not needs_correction:
            return "Validator Agent has determined the cleaning operations are valid."
        lines = ["Validation status: Needs correction"]
        if feedback_target:
            lines.append(f"Feedback target: {feedback_target}")
        if correction_instructions:
            lines.append("")
            lines.append("Correction instructions:")
            lines.append(str(correction_instructions))
        return "\n".join(lines).strip()

    @staticmethod
    def _format_recommender_output_multi(recommender_data) -> str:
        if isinstance(recommender_data, dict) and recommender_data.get("_api_error"):
            return str(recommender_data.get("summary", "") or "").strip() or (
                "API error (HTTP 503): LLM provider unavailable."
            )
        if not isinstance(recommender_data, dict):
            return ""
        parts = []
        summary = str(recommender_data.get("summary", "") or "").strip()
        if summary:
            parts.append(f"Summary:\n{summary}")
        for label, key in [
            ("Violation Instructions", "violation_instructions"),
            ("Imputation Instructions", "imputation_instructions"),
            ("Examples Dirty", "examples_dirty"),
            ("Examples Clean", "examples_clean"),
        ]:
            val = recommender_data.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, list):
                rendered = "\n".join(f"- {json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)}" for v in val)
            elif isinstance(val, dict):
                rendered = "\n".join(f"- {k}: {v}" for k, v in val.items())
            else:
                rendered = str(val)
            parts.append(f"{label}:\n{rendered}")
        return "\n\n".join(parts).strip()

    def _sampler(self, purpose: str, key: str, attempt: int):
        """Seeded random source for one review sample, or the random module when no seed is configured."""
        rng = seeded_random(getattr(self.config, "sampling_seed", None), purpose, key, attempt)
        return rng if rng is not None else random

    def _emit_trace(self, event: TraceEvent):
        if not callable(self.trace_callback):
            return
        try:
            self.trace_callback(event)
        except Exception:
            return

    @staticmethod
    def _attempt_step(prefix: str, label: str, attempt: int, status: str, output: str) -> TraceStep:
        """One numbered step of an agent loop: the id and the title both count from 1."""
        return TraceStep(
            id=f"{prefix}_{attempt + 1}",
            title=f"{label} (attempt {attempt + 1})",
            status=status,
            output=output,
        )

    def _validator_step(
        self,
        prefix: str,
        attempt: int,
        needs_correction: bool,
        feedback_target: str | None,
        correction_instructions: str | None,
    ) -> TraceStep:
        return self._attempt_step(
            prefix,
            "Validator",
            attempt,
            "approved" if not needs_correction else "needs_correction",
            self._format_validator_output(needs_correction, feedback_target, correction_instructions),
        )

    def _emit_step(self, key: str, step: TraceStep):
        """Sends one step to the GUI. Steps that are not kept in the report are emitted this way too."""
        self._emit_trace(
            TraceEvent(column=key, step_id=step.id, title=step.title, status=step.status, output=step.output)
        )

    def _add_step(self, key: str, steps: list[TraceStep], step: TraceStep) -> TraceStep:
        """Keeps a step in the report and sends it to the GUI; returns it, so a later answer can change it."""
        steps.append(step)
        self._emit_step(key, step)
        return step

    def _finish_column(
        self,
        col: str,
        column_type: str,
        steps: list[TraceStep],
        code: str | None,
        *,
        attempts: int,
        cleaned: bool,
        validated: bool,
        output: str,
        already_clean: bool = False,
        status: str = "completed",
        reason: str | None = None,
    ):
        self.cleaning_report[col] = ColumnReport(
            datatype=column_type,
            already_clean=already_clean,
            cleaned=cleaned,
            attempts=attempts,
            generated_code=code,
            cleaning_validated=validated,
            trace_steps=steps,
            reason=reason,
        )
        self._emit_step(col, TraceStep(id="finished", title="Finished", status=status, output=output))

    def _finish_fd(
        self,
        task_key: str,
        target_cols: list[str],
        steps: list[TraceStep],
        code: str | None,
        *,
        attempts: int,
        cleaned: bool,
        validated: bool,
        output: str,
        status: str = "completed",
        reason: str | None = None,
    ):
        self.cleaning_report[task_key] = FDReport(
            target_columns=target_cols,
            cleaned=cleaned,
            attempts=attempts,
            cleaning_validated=validated,
            generated_code=code,
            trace_steps=steps,
            reason=reason,
        )
        self._emit_step(task_key, TraceStep(id="finished", title="Finished", status=status, output=output))

    def _review_sample(
        self,
        dirty_column: pd.Series,
        cleaned_column: pd.Series,
        sampler,
    ) -> tuple[list[dict[str, str]], int]:
        """The rows a reviewer is shown: the configured share of changed rows, then random ones.

        Returns those rows and how many rows changed in total, not just in the sample.
        """
        dirty_column, cleaned_column = align_dirty_cleaned_series(dirty_column, cleaned_column)
        all_indices = list(dirty_column.index)
        changed_indices = [
            idx
            for idx in all_indices
            if str(dirty_column.loc[idx]) != str(cleaned_column.loc[idx])
        ]
        changed_target = max(0, int(getattr(self.config, "sample_size_validator_changed", 0)))
        random_target = max(0, int(getattr(self.config, "sample_size_validator_random", 0)))
        changed_take = min(len(changed_indices), changed_target)
        selected_changed = sampler.sample(changed_indices, changed_take) if changed_take > 0 else []

        remaining = [idx for idx in all_indices if idx not in set(selected_changed)]
        random_take = min(len(remaining), random_target)
        selected_random = sampler.sample(remaining, random_take) if random_take > 0 else []

        sample_indices = selected_changed + selected_random
        # Fallback to total validator sample if no split sizes are configured.
        if not sample_indices:
            sample_size = min(len(dirty_column), max(1, int(self.config.sample_size_validator)))
            sample_indices = sampler.sample(all_indices, sample_size) if sample_size > 0 else []
        o = dirty_column.loc[sample_indices].astype(str).fillna("") if sample_indices else pd.Series(dtype=str)
        cl = cleaned_column.loc[sample_indices].astype(str).fillna("") if sample_indices else pd.Series(dtype=str)
        chg = (o != cl).map(lambda x: "1" if x else "0")
        sample_df = pd.DataFrame({"original": o, "cleaned": cl, "changed": chg})
        total_modified_count = int(
            sum(1 for idx in all_indices if str(dirty_column.loc[idx]) != str(cleaned_column.loc[idx]))
        )
        return sample_df.to_dict("records"), total_modified_count

    def _request_user_validation(
        self,
        col: str,
        attempt: int,
        feedback_target: str | None,
        correction_instructions: str | None,
        dirty_column: pd.Series,
        cleaned_column: pd.Series,
    ) -> tuple[bool, str | None, str | None]:
        if not callable(self.user_validation_callback):
            return True, feedback_target, correction_instructions
        sampler = self._sampler("user_validation", col, attempt)
        try:
            sample_rows, total_modified_count = self._review_sample(dirty_column, cleaned_column, sampler)
            payload = {
                "column": col,
                "attempt": attempt + 1,
                "default_needs_correction": True,
                "default_target": str(feedback_target or "RECOMMENDER"),
                "default_feedback": str(correction_instructions or ""),
                "sample_rows": sample_rows,
                "modified_count": total_modified_count,
            }
            response = self.user_validation_callback(payload)
            if not isinstance(response, dict):
                return True, feedback_target, correction_instructions
            needs_correction = bool(response.get("needs_correction", True))
            selected_target = str(response.get("feedback_target") or feedback_target or "RECOMMENDER")
            feedback = str(response.get("correction_instructions") or correction_instructions or "")
            if selected_target not in ("CODER", "RECOMMENDER"):
                selected_target = "RECOMMENDER"
            return needs_correction, selected_target, feedback
        except Exception:
            return True, feedback_target, correction_instructions

    def _hitl_applies_to_column(self, col: str) -> bool:
        if not getattr(self.config, "human_in_the_loop", False):
            return False
        if getattr(self.config, "hitl_apply_to_all_columns", True):
            return True
        cols = getattr(self.config, "hitl_column_list", None) or []
        return col in cols

    def _hitl_applies_to_fd_task(self, target_cols: list[str]) -> bool:
        # FD / multi-column tasks: HITL not applied (columns-only for now).
        return False

    def _request_hitl_code_review(self, col: str, attempt_idx: int, code: str) -> str:
        if not self._hitl_applies_to_column(col):
            return code
        if not callable(self.hitl_callback):
            return code
        try:
            request_id = f"code::{col}::{attempt_idx}::{time.time()}"
            resp = self.hitl_callback(
                {
                    "kind": "code_review",
                    "request_id": request_id,
                    "column": col,
                    "attempt": attempt_idx + 1,
                    "code": code or "",
                }
            )
            if not isinstance(resp, dict):
                return code
            out = str(resp.get("code", code) or code)
            return out
        except Exception:
            return code

    def _request_hitl_code_review_fd(self, task_key: str, target_cols: list[str], attempt_idx: int, code: str) -> str:
        if not self._hitl_applies_to_fd_task(target_cols):
            return code
        if not callable(self.hitl_callback):
            return code
        try:
            request_id = f"codefd::{task_key}::{attempt_idx}::{time.time()}"
            resp = self.hitl_callback(
                {
                    "kind": "code_review",
                    "request_id": request_id,
                    "column": task_key,
                    "attempt": attempt_idx + 1,
                    "code": code or "",
                }
            )
            if not isinstance(resp, dict):
                return code
            return str(resp.get("code", code) or code)
        except Exception:
            return code

    def _request_hitl_already_clean_review(
        self,
        col: str,
        attempt_idx: int,
        series: pd.Series,
    ) -> tuple[bool, str]:
        """HITL: user confirms 'already clean' or rejects with a short issue description."""
        if not self._hitl_applies_to_column(col):
            return True, ""
        if not callable(self.hitl_callback):
            return True, ""
        sampler = self._sampler("already_clean_review", col, attempt_idx)
        try:
            non_null = series.dropna()
            n = min(12, int(non_null.shape[0]))
            if n <= 0:
                sample_rows: list[dict[str, str]] = []
            else:
                idx_pool = list(non_null.index)
                pick = sampler.sample(idx_pool, n) if len(idx_pool) > n else idx_pool
                sample_rows = []
                for i in pick:
                    o = str(series.loc[i])
                    sample_rows.append({"original": o, "cleaned": o, "changed": "0"})
            request_id = f"ac::{col}::{attempt_idx}::{time.time()}"
            resp = self.hitl_callback(
                {
                    "kind": "already_clean_review",
                    "request_id": request_id,
                    "column": col,
                    "attempt": attempt_idx + 1,
                    "sample_rows": sample_rows,
                }
            )
            if not isinstance(resp, dict):
                return True, ""
            decision = str(resp.get("decision") or "").strip()
            if decision == "confirm":
                return True, ""
            return False, str(resp.get("rejection_reason") or resp.get("errors") or "").strip()
        except Exception:
            return True, ""

    def _request_hitl_validation_review(
        self,
        col: str,
        attempt_idx: int,
        llm_needs_correction: bool,
        feedback_target: str | None,
        correction_instructions: str | None,
        validator_raw: str,
        dirty_column: pd.Series,
        cleaned_column: pd.Series,
    ) -> tuple[bool, str | None, str | None]:
        if not self._hitl_applies_to_column(col):
            return llm_needs_correction, feedback_target, correction_instructions
        if not callable(self.hitl_callback):
            return llm_needs_correction, feedback_target, correction_instructions
        sampler = self._sampler("validation_review", col, attempt_idx)
        try:
            sample_rows, total_modified_count = self._review_sample(dirty_column, cleaned_column, sampler)
            request_id = f"val::{col}::{attempt_idx}::{time.time()}"
            resp = self.hitl_callback(
                {
                    "kind": "validation_review",
                    "request_id": request_id,
                    "column": col,
                    "attempt": attempt_idx + 1,
                    "llm_needs_correction": bool(llm_needs_correction),
                    "llm_feedback_target": feedback_target or "",
                    "llm_correction_instructions": str(correction_instructions or ""),
                    "validator_output": str(validator_raw or ""),
                    "sample_rows": sample_rows,
                    "modified_count": total_modified_count,
                }
            )
            if not isinstance(resp, dict):
                return llm_needs_correction, feedback_target, correction_instructions
            decision = str(resp.get("decision") or "").strip()
            if decision == "validator_ok_agree":
                return False, None, None
            if decision == "validator_ok_disagree":
                ft = str(resp.get("feedback_target") or "RECOMMENDER")
                if ft not in ("CODER", "RECOMMENDER"):
                    ft = "RECOMMENDER"
                msg = str(resp.get("correction_instructions") or "").strip()
                return True, ft, msg or "User disagreed with validation verdict."
            if decision == "feedback_accept":
                return True, feedback_target, correction_instructions
            if decision == "feedback_reject_cleaning_valid":
                return False, None, None
            if decision == "feedback_reject_revise":
                ft = str(resp.get("feedback_target") or feedback_target or "RECOMMENDER")
                if ft not in ("CODER", "RECOMMENDER"):
                    ft = "RECOMMENDER"
                msg = str(resp.get("correction_instructions") or "").strip()
                return True, ft, msg
            # Legacy
            if bool(resp.get("accept", False)):
                return False, None, None
            ft = str(resp.get("feedback_target") or feedback_target or "RECOMMENDER")
            if ft not in ("CODER", "RECOMMENDER"):
                ft = "RECOMMENDER"
            msg = str(resp.get("correction_instructions") or correction_instructions or "")
            return True, ft, msg
        except Exception:
            return llm_needs_correction, feedback_target, correction_instructions

    def _request_hitl_validation_review_fd(
        self,
        task_key: str,
        target_cols: list[str],
        attempt_idx: int,
        llm_needs_correction: bool,
        feedback_target: str | None,
        correction_instructions: str | None,
        validator_raw: str,
        dirty_sample: pd.Series,
        cleaned_sample: pd.Series,
    ) -> tuple[bool, str | None, str | None]:
        if not self._hitl_applies_to_fd_task(target_cols):
            return llm_needs_correction, feedback_target, correction_instructions
        if not callable(self.hitl_callback):
            return llm_needs_correction, feedback_target, correction_instructions
        sampler = self._sampler("validation_review", task_key, attempt_idx)
        try:
            dirty_sample, cleaned_sample = align_dirty_cleaned_series(dirty_sample, cleaned_sample)
            all_indices = list(dirty_sample.index)
            changed_indices = [
                idx for idx in all_indices if str(dirty_sample.loc[idx]) != str(cleaned_sample.loc[idx])
            ]
            changed_target = max(0, int(getattr(self.config, "sample_size_validator_changed", 0)))
            random_target = max(0, int(getattr(self.config, "sample_size_validator_random", 0)))
            changed_take = min(len(changed_indices), changed_target)
            selected_changed = sampler.sample(changed_indices, changed_take) if changed_take > 0 else []
            remaining = [idx for idx in all_indices if idx not in set(selected_changed)]
            random_take = min(len(remaining), random_target)
            selected_random = sampler.sample(remaining, random_take) if random_take > 0 else []
            sample_indices = selected_changed + selected_random
            if not sample_indices:
                sample_size = min(len(dirty_sample), max(1, int(self.config.sample_size_validator)))
                sample_indices = sampler.sample(all_indices, sample_size) if sample_size > 0 else []
            sample_df = pd.DataFrame(
                {
                    "original": dirty_sample.loc[sample_indices].astype(str).fillna("") if sample_indices else [],
                    "cleaned": cleaned_sample.loc[sample_indices].astype(str).fillna("") if sample_indices else [],
                }
            )
            total_modified_count = int(
                sum(1 for idx in all_indices if str(dirty_sample.loc[idx]) != str(cleaned_sample.loc[idx]))
            )
            request_id = f"valfd::{task_key}::{attempt_idx}::{time.time()}"
            resp = self.hitl_callback(
                {
                    "kind": "validation_review",
                    "request_id": request_id,
                    "column": task_key,
                    "attempt": attempt_idx + 1,
                    "llm_needs_correction": bool(llm_needs_correction),
                    "llm_feedback_target": feedback_target or "",
                    "llm_correction_instructions": str(correction_instructions or ""),
                    "validator_output": str(validator_raw or ""),
                    "sample_rows": sample_df.to_dict("records"),
                    "modified_count": total_modified_count,
                }
            )
            if not isinstance(resp, dict):
                return llm_needs_correction, feedback_target, correction_instructions
            if bool(resp.get("accept", False)):
                return False, None, None
            ft = str(resp.get("feedback_target") or feedback_target or "RECOMMENDER")
            if ft not in ("CODER", "RECOMMENDER"):
                ft = "RECOMMENDER"
            msg = str(resp.get("correction_instructions") or correction_instructions or "")
            return True, ft, msg
        except Exception:
            return llm_needs_correction, feedback_target, correction_instructions

    async def _update_token_count(self, agent, input_tokens, output_tokens):
        async with self._token_lock:
            getattr(self.token_usage, agent).add(input_tokens, output_tokens)
            key = _current_task_key.get()
            if key is not None:
                self.per_task_usage.setdefault(key, TokenUsage()).add(input_tokens, output_tokens)
    
    async def _run_column_cleaning_async(self, df: pd.DataFrame, col: str, column_profile: ColumnProfile) -> tuple[str, pd.Series | None, str]:
        """Manages the cleaning and verification workflow for a single column. This is the core interaction loop."""
        # Every model call made from here on is this column's, for the token statistics.
        _current_task_key.set(col)
        column_type = column_profile.semantic_type
        recommender_history = None
        coder_history = None
        validator_history = None
        recommender_data = None
        cleaned_column = None 
        final_code_str = ""
        trace_steps: list[TraceStep] = []
        recommender_api_failure_summary: str | None = None
        # 1. Run multi-agent cleaning loop. Start with Recommender Agent to generate instructions
        feedback_target = 'RECOMMENDER'
        for attempt in range(self.config.max_cleaning_attempts):
            self._check_cancelled()
            if feedback_target == 'RECOMMENDER':
                already_clean, recommender_data, new_recommender_history = await self.recommender_agent.generate_recommendations_async(
                    col, column_profile, messages=recommender_history)
                recommender_history = new_recommender_history
                self._add_step(col, trace_steps, self._attempt_step(
                    "recommender",
                    "Recommender",
                    attempt,
                    self._recommender_trace_status(recommender_data, already_clean=already_clean),
                    self._format_recommender_output(recommender_data, already_clean=already_clean),
                ))
                if isinstance(recommender_data, dict) and recommender_data.get("_api_error"):
                    recommender_api_failure_summary = str(
                        recommender_data.get("summary", "") or ""
                    ).strip() or None
                    if attempt < self.config.max_cleaning_attempts - 1:
                        continue
                    break
                if already_clean:
                    if self._hitl_applies_to_column(col):
                        review_step = self._add_step(col, trace_steps, TraceStep(
                            id="validator_already_clean",
                            title="Validator (already clean check)",
                            status="pending_hitl",
                            output="Confirm that this column looks clean, or reject and describe what is wrong.",
                        ))
                        user_confirms, rejection = await asyncio.to_thread(
                            self._request_hitl_already_clean_review,
                            col,
                            attempt,
                            df[col],
                        )
                        if not user_confirms:
                            review_step.status = "rejected"
                            review_step.output = str(rejection or "User rejected already-clean assessment.")
                            self._emit_step(col, review_step)
                            fb = (str(rejection).strip() or "User indicates the column still has issues that need cleaning.").strip()
                            recommender_history.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "User review: the column should NOT be treated as already clean.\n"
                                        f"Issues reported: {fb}\n"
                                        "Provide updated cleaning instructions and examples."
                                    ),
                                }
                            )
                            continue
                        review_step.status = "approved"
                        review_step.output = "User confirmed: values look acceptable without cleaning."
                        self._emit_step(col, review_step)
                    msg = f"[{col}] Recommender Agent determined column is already clean."
                    self._finish_column(
                        col, column_type, trace_steps, final_code_str,
                        attempts=attempt + 1,
                        cleaned=False,
                        validated=True if self._hitl_applies_to_column(col) else False,
                        already_clean=True,
                        output="Column is already clean.",
                    )
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
            _coder_status = "ok" if cleaned_column is not None else (
                "api_unavailable" if isinstance(final_code_str, str) and final_code_str.startswith("API error (HTTP 503 Service Unavailable)")
                else "failed"
            )
            coder_step = self._add_step(col, trace_steps, self._attempt_step(
                "coder", "Coder", attempt, _coder_status, self._trace_text(final_code_str)
            ))
            if cleaned_column is not None and self._hitl_applies_to_column(col):
                reviewed_code = await asyncio.to_thread(
                    self._request_hitl_code_review,
                    col,
                    attempt,
                    final_code_str or "",
                )
                if self.verbose:
                    print(
                        f"[PIPELINE][post_code_review] column={col} attempt={attempt + 1} "
                        f"ready_for_validation={'yes' if cleaned_column is not None else 'no'}"
                    )
                if reviewed_code.strip() != (final_code_str or "").strip():
                    exec_result = await LLMCodingAgent._execute_code_async(reviewed_code, df, col)
                    if isinstance(exec_result, pd.Series):
                        cleaned_column = exec_result
                        final_code_str = reviewed_code
                        coder_step.status = "ok"
                        coder_step.output = self._trace_text(final_code_str)
                        self._emit_step(col, coder_step)
                    else:
                        self._emit_step(col, TraceStep(
                            id=f"coder_user_edit_{attempt + 1}",
                            title="User code review",
                            status="needs_correction",
                            output=f"Edited code failed to execute: {exec_result!s}. Using coder output.",
                        ))
            # 3. If Coding Agent could not generate valid code to clean column, send feedback to Recommender to provide better instructions
            if cleaned_column is None: 
                if attempt == self.config.max_cleaning_attempts -1:
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
                self._finish_column(
                    col, column_type, trace_steps, final_code_str,
                    attempts=attempt + 1, cleaned=True, validated=False,
                    output="Column cleaned successfully.",
                )
                return col, cleaned_column, msg

            if column_type in ("INTEGER", "FLOAT", "BOOLEAN"):
                msg = f"[{col}] Successfully cleaned."
                self._finish_column(
                    col, column_type, trace_steps, final_code_str,
                    attempts=attempt + 1, cleaned=True, validated=False,
                    output="Column cleaned successfully (validation skipped for numeric/boolean type).",
                )
                return col, cleaned_column, msg

            # Validator entry log removed from standard output; tracing is handled via structured events.
            if self.config.enable_user_validation:
                self._emit_step(col, self._attempt_step(
                    "validator", "Validator", attempt, "needs_user_validation",
                    "Validation requires user review. Click this step to provide feedback.",
                ))
                needs_correction, feedback_target, correction_instructions = await asyncio.to_thread(
                    self._request_user_validation,
                    col,
                    attempt,
                    "RECOMMENDER",
                    "",
                    df[col],
                    cleaned_column,
                )
                self._add_step(col, trace_steps, self._validator_step(
                    "validator", attempt, needs_correction, feedback_target, correction_instructions
                ))
                if not needs_correction:
                    msg = f"[{col}] Successfully cleaned and user validated."
                    self._finish_column(
                        col, column_type, trace_steps, final_code_str,
                        attempts=attempt + 1, cleaned=True, validated=True,
                        output="Column cleaned and validated by user.",
                    )
                    return col, cleaned_column, msg
                if attempt == self.config.max_cleaning_attempts - 1:
                    break
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
                continue

            last_attempt = attempt == (self.config.max_cleaning_attempts - 1)
            needs_correction, feedback_target, correction_instructions, new_validator_history, validator_raw = await self.validation_agent.validate_async(
                col, df[col], cleaned_column, column_type, messages=validator_history, last_attempt=last_attempt,
                attempt=attempt)
            validator_history = new_validator_history
            if self._hitl_applies_to_column(col) and not self.config.enable_user_validation:
                self._emit_step(col, self._attempt_step(
                    "validator", "Validator", attempt, "pending_hitl",
                    "Validation result is ready for human review. Open this validator step to approve or revise.",
                ))
                needs_correction, feedback_target, correction_instructions = await asyncio.to_thread(
                    self._request_hitl_validation_review,
                    col,
                    attempt,
                    needs_correction,
                    feedback_target,
                    correction_instructions,
                    validator_raw,
                    df[col],
                    cleaned_column,
                )
            if correction_instructions == "__VALIDATOR_FAILURE_LEAVE_UNCLEANED__":
                msg = f"[{col}] Validation unavailable; leaving column uncleaned."
                self._finish_column(
                    col, column_type, trace_steps, final_code_str,
                    attempts=attempt + 1, cleaned=False, validated=False,
                    output="Validation unavailable; column left uncleaned.",
                    status="failed",
                    reason="Validator failed and strategy is leave_uncleaned",
                )
                return col, None, msg
            if correction_instructions == "__VALIDATOR_FAILURE_ASK_USER__" and not self.config.enable_user_validation:
                self._emit_step(col, self._attempt_step(
                    "validator", "Validator", attempt, "needs_user_validation",
                    "Validator failed. User feedback is required.",
                ))
                needs_correction, feedback_target, correction_instructions = await asyncio.to_thread(
                    self._request_user_validation,
                    col,
                    attempt,
                    "RECOMMENDER",
                    "Validator failed to validate. Please provide final feedback.",
                    df[col],
                    cleaned_column,
                )
            self._add_step(col, trace_steps, self._validator_step(
                "validator", attempt, needs_correction, feedback_target, correction_instructions
            ))
            # 5. If validator approves cleaned column, return column. Otherwise provide feedback to corresponding Agent
            if not needs_correction:
                msg = f"[{col}] Successfully cleaned and validated."
                self._finish_column(
                    col, column_type, trace_steps, final_code_str,
                    attempts=attempt + 1, cleaned=True, validated=True,
                    output="Column cleaned and validated.",
                )
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
        msg = f"[{col}] FAILED cleaning after {self.config.max_cleaning_attempts} attempts."
        self._finish_column(
            col, column_type, trace_steps, final_code_str,
            attempts=self.config.max_cleaning_attempts, cleaned=False, validated=False,
            output=recommender_api_failure_summary or "Cleaning failed after maximum attempts.",
            status="failed",
        )
        return col, None, msg

    async def _run_multi_col_cleaning_async(self, df: pd.DataFrame, task_info: MultiColumnTask) -> tuple[list[str], pd.DataFrame | None, str]:
        """Manages the cleaning and verification workflow for a multi-column operations. This is the core interaction loop."""          
        recommender_history = None
        coder_history = None
        validator_history = None
        recommender_data = None
        cleaned_targets = None 
        _final_code_str = ""
        target_cols = task_info.target_columns
        task_key = task_info.verbose_key
        # Every model call made from here on belongs to this dependency task.
        _current_task_key.set(task_key)
        trace_steps: list[TraceStep] = []
        fd_recommender_api_failure_summary: str | None = None
        # 1. Run multi-agent cleaning loop. Start with RecommenderAgent
        feedback_target = 'RECOMMENDER'
        for attempt in range(self.config.max_multi_col_attempts):
            self._check_cancelled()
            if feedback_target == 'RECOMMENDER':
                recommender_data, new_recommender_history = await self.recommender_agent.generate_recommendations_multi_col_async( 
                    df, task_info, messages=recommender_history)
                recommender_history = new_recommender_history
                self._add_step(task_key, trace_steps, self._attempt_step(
                    "fd_recommender",
                    "Recommender",
                    attempt,
                    self._recommender_trace_status(recommender_data, already_clean=False),
                    self._format_recommender_output_multi(recommender_data),
                ))
                if isinstance(recommender_data, dict) and recommender_data.get("_api_error"):
                    fd_recommender_api_failure_summary = str(
                        recommender_data.get("summary", "") or ""
                    ).strip() or None
                    if attempt < self.config.max_multi_col_attempts - 1:
                        continue
                    break
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
            _fd_coder_status = "ok" if cleaned_targets is not None else (
                "api_unavailable" if isinstance(_final_code_str, str) and _final_code_str.startswith("API error (HTTP 503 Service Unavailable)")
                else "failed"
            )
            fd_coder_step = self._add_step(task_key, trace_steps, self._attempt_step(
                "fd_coder", "Coder", attempt, _fd_coder_status, self._trace_text(_final_code_str)
            ))
            if cleaned_targets is not None and self._hitl_applies_to_fd_task(target_cols):
                reviewed_code = self._request_hitl_code_review_fd(task_key, target_cols, attempt, _final_code_str or "")
                if reviewed_code.strip() != (_final_code_str or "").strip():
                    exec_result = await LLMCodingAgent._execute_code_async(reviewed_code, df, target_cols)
                    if isinstance(exec_result, pd.DataFrame):
                        cleaned_targets = exec_result
                        _final_code_str = reviewed_code
                        fd_coder_step.status = "ok"
                        fd_coder_step.output = self._trace_text(_final_code_str)
                        self._emit_step(task_key, fd_coder_step)
                    else:
                        self._emit_step(task_key, TraceStep(
                            id=f"fd_coder_user_edit_{attempt + 1}",
                            title="User code review",
                            status="needs_correction",
                            output=f"Edited code failed to execute: {exec_result!s}. Using coder output.",
                        ))
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
                self._finish_fd(
                    task_key, target_cols, trace_steps, _final_code_str,
                    attempts=attempt + 1, cleaned=True, validated=False,
                    output="FD task cleaned successfully.",
                )
                return target_cols, cleaned_targets, msg

            if self.config.enable_user_validation:
                self._emit_step(task_key, self._attempt_step(
                    "fd_validator", "Validator", attempt, "needs_user_validation",
                    "Validation requires user review. Click this step to provide feedback.",
                ))
                compared_cols = [c for c in target_cols if c in df.columns and c in cleaned_targets.columns]
                if not compared_cols:
                    compared_cols = [df.columns[0]]
                dirty_sample = df[compared_cols].astype(str).apply(lambda r: ", ".join(r.tolist()), axis=1)
                cleaned_sample = cleaned_targets[compared_cols].astype(str).apply(lambda r: ", ".join(r.tolist()), axis=1)
                needs_correction, feedback_target, correction_instructions = await asyncio.to_thread(
                    self._request_user_validation,
                    task_key,
                    attempt,
                    "RECOMMENDER",
                    "",
                    dirty_sample,
                    cleaned_sample,
                )
                self._add_step(task_key, trace_steps, self._validator_step(
                    "fd_validator", attempt, needs_correction, feedback_target, correction_instructions
                ))
                if not needs_correction:
                    msg = f"[{task_key}] Successfully cleaned and user validated."
                    self._finish_fd(
                        task_key, target_cols, trace_steps, _final_code_str,
                        attempts=attempt + 1, cleaned=True, validated=True,
                        output="FD task cleaned and validated by user.",
                    )
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
                continue

            last_attempt = attempt == (self.config.max_multi_col_attempts - 1)
            needs_correction, feedback_target, correction_instructions, new_validator_history, validator_raw = await self.validation_agent.validate_multi_col_async(
                dirty_targets=df[target_cols],
                cleaned_targets=cleaned_targets,
                task_info=task_info,
                messages=validator_history,
                last_attempt=last_attempt,
                attempt=attempt)
            validator_history = new_validator_history
            if self._hitl_applies_to_fd_task(target_cols) and not self.config.enable_user_validation:
                self._emit_step(task_key, self._attempt_step(
                    "fd_validator", "Validator", attempt, "pending_hitl",
                    "Validation result is ready for human review. Open this validator step to approve or revise.",
                ))
                compared_cols = [c for c in target_cols if c in df.columns and c in cleaned_targets.columns]
                if not compared_cols:
                    compared_cols = [df.columns[0]]
                dirty_sample = df[compared_cols].astype(str).apply(lambda r: ", ".join(r.tolist()), axis=1)
                cleaned_sample = cleaned_targets[compared_cols].astype(str).apply(lambda r: ", ".join(r.tolist()), axis=1)
                needs_correction, feedback_target, correction_instructions = await asyncio.to_thread(
                    self._request_hitl_validation_review_fd,
                    task_key,
                    target_cols,
                    attempt,
                    needs_correction,
                    feedback_target,
                    correction_instructions,
                    validator_raw,
                    dirty_sample,
                    cleaned_sample,
                )
            if correction_instructions == "__VALIDATOR_FAILURE_LEAVE_UNCLEANED__":
                msg = f"[{task_key}] Validation unavailable; leaving FD task uncleaned."
                self._finish_fd(
                    task_key, target_cols, trace_steps, _final_code_str,
                    attempts=attempt + 1, cleaned=False, validated=False,
                    output="Validation unavailable; FD task left uncleaned.",
                    status="failed",
                    reason="Validator failed and strategy is leave_uncleaned",
                )
                return target_cols, None, msg
            if correction_instructions == "__VALIDATOR_FAILURE_ASK_USER__" and not self.config.enable_user_validation:
                self._emit_step(task_key, self._attempt_step(
                    "fd_validator", "Validator", attempt, "needs_user_validation",
                    "Validator failed. User feedback is required.",
                ))
                compared_cols = [c for c in target_cols if c in df.columns and c in cleaned_targets.columns]
                if not compared_cols:
                    compared_cols = [df.columns[0]]
                dirty_sample = df[compared_cols].astype(str).apply(lambda r: ", ".join(r.tolist()), axis=1)
                cleaned_sample = cleaned_targets[compared_cols].astype(str).apply(lambda r: ", ".join(r.tolist()), axis=1)
                needs_correction, feedback_target, correction_instructions = await asyncio.to_thread(
                    self._request_user_validation,
                    task_key,
                    attempt,
                    "RECOMMENDER",
                    "Validator failed to validate FD output. Please provide final feedback.",
                    dirty_sample,
                    cleaned_sample,
                )
            self._add_step(task_key, trace_steps, self._validator_step(
                "fd_validator", attempt, needs_correction, feedback_target, correction_instructions
            ))
            # 5. If validator approves cleaned columns, return columns. Otherwise provide feedback to corresponding Agent
            if not needs_correction:
                msg = f"[{task_key}] Successfully cleaned and validated."
                self._finish_fd(
                    task_key, target_cols, trace_steps, _final_code_str,
                    attempts=attempt + 1, cleaned=True, validated=True,
                    output="FD task cleaned and validated.",
                )
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
        self._finish_fd(
            task_key, target_cols, trace_steps, _final_code_str,
            attempts=self.config.max_multi_col_attempts, cleaned=False, validated=False,
            output=fd_recommender_api_failure_summary or "FD task failed after maximum attempts.",
            status="failed",
        )
        return target_cols, None, msg


