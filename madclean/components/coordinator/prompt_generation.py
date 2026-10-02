import random
from collections import Counter
import pandas as pd
from typing import Callable, cast
# Local imports
from madclean.components.coordinator.prompts import *
from madclean.utils.helpers import (
    format_list_for_prompt,
    format_prompt_template,
    align_dirty_cleaned_series,
    align_dirty_cleaned_dataframe,
)
from madclean.components.domain.schema import ColumnProfile, OutlierResult, MultiColumnTask, FDResult

class PromptGeneration:
    "Instantiates all prompt templates with the provided data for downstream LLM use."""
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.context_formatters: dict[str, Callable] = {
            "outlier_data": self._format_outlier_data
        }
    
    def create_prompt_recommender(
        self,
        col: str,
        profile: ColumnProfile,
        *,
        user_constraints: str = "",
        labeled_examples: str = "",
    ) -> str:
        column_type = profile.semantic_type
        column_sample = profile.sample
        # 1. Get base prompt template
        recommender_prompt_template = RECOMMENDATION_PROMPT_TEMPLATES.get(column_type)
        # 2. Generate context blocks
        additional_context = []
        # Check known fields of ColumnProfiler (e.g., OutlierDetection)
        if profile.outlier_data:
            formatted_block = self._format_outlier_data(profile.outlier_data)
            if formatted_block:
                additional_context.append(formatted_block)
        # Check metadata for extensions
        for key, data in profile.metadata.items():
            if key in self.context_formatters and data:
                formatted_block = self.context_formatters[key](data)
                if formatted_block:
                    additional_context.append(formatted_block)
        # 3. Assemble final prompt
        context_str = "\n" + "\n\n".join(additional_context).strip() + "\n" if additional_context else ""
        uc = (user_constraints or "").strip()
        le = (labeled_examples or "").strip()
        final_prompt = format_prompt_template(
            recommender_prompt_template,
            column_name=col,
            column_sample=column_sample,
            additional_context=context_str,
            user_constraints=uc if uc else "(none)",
            labeled_examples=le if le else "(none)",
        )
        return final_prompt

    def _format_outlier_data(self, outlier_data: OutlierResult) -> str:
        """Formatter for outliers data."""
        if not outlier_data:
            return ""
        # 1. Format outliers into string
        outliers="[" + ", ".join(f"{outl} ({count})" for outl, count in outlier_data.outliers) + "]" 
        formatted_context = []
        for row_list in outlier_data.context:
            formatted_context.append(",".join(format_list_for_prompt(row_list)))
        context_str = "\n".join(formatted_context).strip()
        # 2. Assemble outlier prompt block
        outlier_prompt_template = OUTLIER_PROMPT_TEMPLATE
        final_prompt = format_prompt_template(
                outlier_prompt_template,
                median=outlier_data.median,
                mad=outlier_data.mad,
                outliers=outliers,
                context_rows=context_str)
        return final_prompt

    def create_prompt_coding(self, col: str, column_type: str, recommender_data: dict, allowed_packages: str) -> str:
        """Creates the final prompt for the LLM based on the column's type and sample."""
        def format_data(data: list) -> str:
            if data is not None:
                data_with_lines = [f"- {d}" for d in data]
                return "\n".join(data_with_lines)
            return ""
        coding_prompt_template = CODING_PROMPT_TEMPLATE
        final_prompt = format_prompt_template(coding_prompt_template,
                                                     column_name=col,
                                                     column_type=column_type,
                                                     summary=recommender_data['summary'] if recommender_data['summary'] is not None else "",
                                                     error_types=format_data(recommender_data['error_types']),
                                                     examples_clean=format_data(recommender_data['examples_clean']),
                                                     examples_dirty=format_data(recommender_data['examples_dirty']),
                                                     cleaning_instructions=format_data(recommender_data['cleaning_instructions']),
                                                     allowed_packages=allowed_packages)
        return final_prompt

    def create_prompt_validation(self, col: str, 
            dirty_series: pd.Series, cleaned_series: pd.Series, 
            column_type: str, last_attempt: bool = False,
            max_sample_size: int = 150,
            random_sample_size: int = 60,
            changed_sample_size: int = 90,
            rng: random.Random | None = None) -> str:
        PROMPT_MAP = {
        "DATETIME": "DATETIME",
        "BOOLEAN": "BOOLEAN",
        "INTEGER": "NUMERIC",
        "FLOAT": "NUMERIC",
        "DIRTY_INTEGER": "DIRTY_NUMERIC",
        "DIRTY_FLOAT": "DIRTY_NUMERIC",
        "NAMED_ENTITY": "STRING",
        "DISCRETE_STRING": "STRING",
        "COLLECTION": "STRING",
        "DELIMITED_STRING": "STRING",
        "NATURAL_LANGUAGE_TEXT": "NLT"
        }
        dirty_series, cleaned_series = align_dirty_cleaned_series(dirty_series, cleaned_series)
        # 1. Describe the whole column. changed_sample_size only sizes the human review sample now.
        column_comparison_str = self._column_overview(
            dirty_series, cleaned_series,
            max_rewrites=max_sample_size,
            unchanged_sample_size=random_sample_size,
            rng=rng,
        )
        # 2. Add last attempt message to Validator Agent
        last_attempt_msg = (
            "\nThis is the final validation attempt. "
            "If the column maintains the same dominant format and style as the original column (no undesired changes such as casing, patterns or structure), approve the column as clean (set 'needs_correction' to False) even if some minor errors remain. "
            "Since partial correction without undesired changes is preferred over reverting to the original column.\n"
        )
        # 3. Instantiate prompt template
        prompt_type = PROMPT_MAP.get(column_type)
        prompt_template = VALIDATION_PROMPT_TEMPLATES.get(prompt_type)
        final_prompt = format_prompt_template(
            prompt_template,
            last_attempt_msg=last_attempt_msg if last_attempt else "",
            column_name=col,
            column_comparison_sample=column_comparison_str
        )
        return final_prompt

    OVERVIEW_TOP_VALUES = 30
    OVERVIEW_VALUE_WIDTH = 80

    @staticmethod
    def _is_missing(value) -> bool:
        return pd.api.types.is_scalar(value) and bool(pd.isna(value))

    @classmethod
    def _cell_changed(cls, dirty_value, cleaned_value) -> bool:
        """Two missing values are equal; otherwise values compare as text, so 12 and "12" are unchanged."""
        dirty_missing, cleaned_missing = cls._is_missing(dirty_value), cls._is_missing(cleaned_value)
        if dirty_missing or cleaned_missing:
            return dirty_missing != cleaned_missing
        return str(dirty_value) != str(cleaned_value)

    @classmethod
    def _overview_text(cls, value) -> str | None:
        """A value as the overview prints it: None for missing, long values cut with an ellipsis."""
        if cls._is_missing(value):
            return None
        if hasattr(value, "item") and pd.api.types.is_scalar(value):
            value = value.item()
        text = str(value)
        if len(text) > cls.OVERVIEW_VALUE_WIDTH:
            text = text[:cls.OVERVIEW_VALUE_WIDTH] + "…"
        return text

    @staticmethod
    def _shown(text: str | None, quoted: bool) -> str:
        if text is None:
            return "<empty>"
        return f'"{text}"' if quoted else text

    @staticmethod
    def _by_count(counter: Counter) -> list:
        """Most frequent first; ties keep the order in which the values first appear."""
        return sorted(counter.items(), key=lambda item: -item[1])

    def _column_overview(
        self,
        dirty_series: pd.Series,
        cleaned_series: pd.Series,
        *,
        max_rewrites: int,
        unchanged_sample_size: int,
        rng: random.Random | None = None,
    ) -> str:
        """Whole-column facts for the validator: counts, the change share, each distinct rewrite once with its
        row count, and a sample of unchanged values. Repeated rewrites are one line, so they cannot look like
        the column's norm the way a sample of mostly changed rows did."""
        sampler = rng if rng is not None else random
        rows = len(dirty_series)
        before = [self._overview_text(v) for v in dirty_series.tolist()]
        after = [self._overview_text(v) for v in cleaned_series.tolist()]
        rewrites: Counter = Counter()
        unchanged: dict[str, None] = {}
        emptied = 0
        for dirty_value, cleaned_value, b, a in zip(dirty_series.tolist(), cleaned_series.tolist(), before, after):
            if not self._cell_changed(dirty_value, cleaned_value):
                if b is not None:
                    unchanged.setdefault(b, None)
                continue
            rewrites[(b, a)] += 1
            if a is None:
                emptied += 1
        changed = sum(rewrites.values())
        share = (100.0 * changed / rows) if rows else 0.0
        lines = [
            f"Rows: {rows}",
            f"Filled cells before cleaning: {sum(v is not None for v in before)}; "
            f"after cleaning: {sum(v is not None for v in after)}",
            f"{changed} of {rows} cells changed ({share:.1f}%)",
            f"{emptied} cells emptied",
        ]
        for label, values in (("BEFORE", before), ("AFTER", after)):
            lines += ["", f"MOST FREQUENT VALUES {label} CLEANING (value (count))"]
            top = self._by_count(Counter(values))[:self.OVERVIEW_TOP_VALUES]
            lines += [f"- {self._shown(value, quoted=False)} ({count})" for value, count in top]
        if rewrites:
            ordered = self._by_count(rewrites)
            shown = ordered[:max(0, max_rewrites)]
            lines += ["", "DISTINCT REWRITES (original → cleaned (rows)), most rows first"]
            lines += [
                f"{self._shown(old, quoted=True)} → {self._shown(new, quoted=True)} ({count} rows)"
                for (old, new), count in shown
            ]
            rest = ordered[len(shown):]
            if rest:
                lines.append(
                    f"… and {len(rest)} more distinct rewrites covering {sum(count for _, count in rest)} rows"
                )
        unchanged_values = list(unchanged)
        take = min(len(unchanged_values), max(0, unchanged_sample_size))
        if take:
            lines += ["", "UNCHANGED VALUES (a sample)"]
            lines += [self._shown(value, quoted=True) for value in sampler.sample(unchanged_values, take)]
        return "\n".join(lines)

    # ===== Multi-column operations =====
    def create_prompt_recommender_multi_col(
        self,
        df: pd.DataFrame,
        task_info: MultiColumnTask,
        max_sample_size: int = 40,
        *,
        user_constraints: str = "",
        labeled_examples: str = "",
    ) -> str:
        """Enables multiple multi-column cleaning components to use this function."""
        task_type = task_info.task_type
        if task_type == 'FD':
            return self._create_prompt_recommender_fd(
                df, task_info, max_sample_size,
                user_constraints=user_constraints,
                labeled_examples=labeled_examples,
            )
        else:
            raise ValueError(f"Unsupported multi-column task type: {task_type}")

    def create_prompt_coding_multi_col(self, task_info: MultiColumnTask, recommender_data: dict, allowed_packages: str) -> str:
        """Enables multiple multi-column cleaning components to use this function."""
        task_type = task_info.task_type
        if task_type == 'FD':
            return self._create_prompt_coding_fd(task_info, recommender_data, allowed_packages)
        else:
            raise ValueError(f"Unsupported multi-column task type: {task_type}")

    def create_prompt_validation_multi_col(self, 
                    dirty_target: pd.DataFrame, cleaned_target: pd.DataFrame, 
                    task_info: MultiColumnTask, last_attempt: bool, 
                    max_sample_size: int = 150,
                    random_sample_size: int = 60,
                    changed_sample_size: int = 90,
                    rng: random.Random | None = None) -> str:
        """Enables multiple multi-column cleaning components to use this function."""
        task_type = task_info.task_type
        comparison_str = self._generate_multi_col_comparison(
            dirty_target, cleaned_target, max_sample_size, random_sample_size, changed_sample_size, rng)
        if task_type == 'FD':
            return self._create_prompt_validation_fd(task_info, last_attempt, comparison_str)
        else:
            raise ValueError(f"Unsupported multi-column task type: {task_type}")

    def _generate_multi_col_comparison(
        self,
        dirty_target: pd.DataFrame,
        cleaned_target: pd.DataFrame,
        max_sample_size: int,
        random_sample_size: int,
        changed_sample_size: int,
        rng: random.Random | None = None,
    ) -> str:
        """
        Generates comparision string for any number of columns.
        Format: dirty_col1, dirty_col2 => clean_col1, clean_col2
        """
        sampler = rng if rng is not None else random
        dirty_target, cleaned_target = align_dirty_cleaned_dataframe(dirty_target, cleaned_target)
        # 1. Sample dataset
        all_indices = list(dirty_target.index)
        changed_indices: list = []
        for idx in all_indices:
            dirty_vals = dirty_target.loc[idx].tolist()
            cleaned_vals = cleaned_target.loc[idx].tolist()
            changed = False
            for a, b in zip(dirty_vals, cleaned_vals):
                if pd.isna(a) and pd.isna(b):
                    continue
                if str(a) != str(b):
                    changed = True
                    break
            if changed:
                changed_indices.append(idx)
        changed_take = min(len(changed_indices), max(0, changed_sample_size))
        selected_changed = sampler.sample(changed_indices, changed_take) if changed_take > 0 else []
        remaining = [idx for idx in all_indices if idx not in set(selected_changed)]
        random_take = min(len(remaining), max(0, random_sample_size))
        selected_random = sampler.sample(remaining, random_take) if random_take > 0 else []
        sample_indices = selected_changed + selected_random
        if not sample_indices:
            sample_size = min(len(all_indices), max_sample_size)
            sample_indices = sampler.sample(all_indices, sample_size) if sample_size > 0 else []
        if max_sample_size > 0 and len(sample_indices) > max_sample_size:
            sample_indices = sample_indices[:max_sample_size]
        # 2. Format values to correct string format
        lines= []
        for i in sample_indices:
            dirty_vals = dirty_target.loc[i].tolist()
            cleaned_vals = cleaned_target.loc[i].tolist()
            def fmt(v): return "NaN" if pd.isna(v) else str(v).replace('"', "'")
            # 3. Combine columns into 1 string per row
            d_str = ", ".join([f'"{fmt(v)}"' for v in dirty_vals])
            c_str = ", ".join([f'"{fmt(v)}"' for v in cleaned_vals])
            lines.append(f'{d_str} => {c_str}')
        return "\n".join(lines)   
    
    # ===== FD implementation =====
    def _create_prompt_recommender_fd(
        self,
        df: pd.DataFrame,
        task_info: MultiColumnTask,
        max_sample_size: int,
        *,
        user_constraints: str = "",
        labeled_examples: str = "",
    ) -> str:
        def _format_violations_for_prompt(violations: list[dict]) -> str:
            formatted_output = []
            for violation in violations:
                lhs_val = violation['lhs']
                conflicting_rhs = violation['rhs_conflicts']
                context_rows = violation['context']
                violation_header = f"{{'lhs': '{lhs_val}', 'rhs_conflicts': {conflicting_rhs}}}"
                formatted_output.append(violation_header)
                for row_list in context_rows:
                    formatted_output.append(",".join(format_list_for_prompt(row_list)))
            return "\n".join(formatted_output).strip()
        # 1. Unpack data
        fd_data = cast(FDResult, task_info.data)
        lhs, rhs = fd_data.lhs, fd_data.rhs
        fd_violation_data = fd_data.violation_data if fd_data.violation_data else {}
        fd_imputation_data = fd_data.imputation_data if fd_data.imputation_data else {}
        violation_count = fd_violation_data.get('count', 0) # use fd_data.violation count
        imputation_count = fd_imputation_data.get('count', 0)
        # 2. Create sample
        fd_pair_df = df[[lhs, rhs]]
        sample_size = min(len(fd_pair_df), max_sample_size)
        fd_pair_sample = fd_pair_df.sample(n=sample_size, random_state=42)
        # pandas ends lines the operating system's way; naming \r\n, what the paper's Windows runs
        # sent, keeps this prompt identical on every machine.
        fd_pair_sample_str = fd_pair_sample.to_csv(index=False, lineterminator="\r\n").strip()
        # 3. Instantiate prompt template
        template = FD_RECOMMENDATION_PROMPT_TEMPLATE
        uc = (user_constraints or "").strip()
        le = (labeled_examples or "").strip()
        fmt_kwargs = dict(
            lhs=lhs,
            rhs=rhs,
            fd_pair_sample=fd_pair_sample_str,
            imputable_count=imputation_count,
            violation_count=violation_count,
            user_constraints=uc if uc else "(none)",
            labeled_examples=le if le else "(none)",
        )
        if violation_count > 0:
            header_str = ",".join(df.columns)
            violations = fd_violation_data.get('violations', [])
            violations_str = _format_violations_for_prompt(violations)
            return format_prompt_template(
                template,
                **fmt_kwargs,
                column_header=header_str,
                violations=violations_str,
            )
        else:
            return format_prompt_template(
                template,
                **fmt_kwargs,
                column_header="-",
                violations="No violations",
            )

    def _create_prompt_coding_fd(self, task_info: MultiColumnTask, recommender_data: dict, allowed_packages: str) -> str:
        fd_data = cast(FDResult, task_info.data)
        return format_prompt_template(
            FD_CODING_PROMPT_TEMPLATE,
            lhs=fd_data.lhs,
            rhs=fd_data.rhs,
            summary=recommender_data['summary'],
            violation_fix_instructions=recommender_data['violation_instructions'],
            missing_values_imputation_instructions=recommender_data['imputation_instructions'],
            allowed_packages=allowed_packages)
    
    def _create_prompt_validation_fd(self, task_info: MultiColumnTask, 
                                    last_attempt: bool, comparison_str: str) -> str:
        fd_data = cast(FDResult, task_info.data)
        lhs, rhs = fd_data.lhs, fd_data.rhs
        last_attempt_msg = (
            "\nThis is the final validation attempt. "
            "If no undesired changes have occured, approve the columns as clean (set 'needs_correction' to False) even if some minor errors remain. "
            "Since partial correction without undesired changes is preferred over reverting to the original column.\n")
        return format_prompt_template(
            FD_VALIDATION_PROMPT_TEMPLATE,
            lhs=lhs,
            rhs=rhs,
            last_attempt_msg=last_attempt_msg if last_attempt else "",
            fd_comparison_sample=comparison_str)