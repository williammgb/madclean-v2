import random
import pandas as pd
from typing import Callable, cast
# Local imports
from madclean.components.coordinator.prompts import *
from madclean.utils.helpers import format_list_for_prompt
from madclean.components.domain.schema import ColumnProfile, OutlierResult, MultiColumnTask, FDResult

class PromptGeneration:
    "Instantiates all prompt templates with the provided data for downstream LLM use."""
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.context_formatters: dict[str, Callable] = {
            "outlier_data": self._format_outlier_data
        }
    
    def create_prompt_recommender(self, col: str, profile: ColumnProfile) -> str:
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
        final_prompt = recommender_prompt_template.format(
            column_name=col,
            column_sample=column_sample,
            additional_context=context_str
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
        final_prompt = outlier_prompt_template.format(
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
        final_prompt = coding_prompt_template.format(column_name=col,
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
            max_sample_size: int = 150) -> str:
        PROMPT_MAP = {
        "DATETIME": "DATETIME",
        "BOOLEAN": "BOOLEAN",
        "INTEGER": "NUMERIC",
        "FLOAT": "NUMERIC",
        "DIRTY_INTEGER": "DIRTY_NUMERIC",
        "DIRTY_FLOAT": "DIRTY_NUMERIC",
        "NAMED_ENTITY": "STRING",
        "DISCRETE_STRING": "STRING",
        "NATURAL_LANGUAGE_TEXT": "NLT"
        }
        # 1. Configure prompt based on semantic type
        sample_size = min(len(dirty_series), max_sample_size)
        sample_indices = random.sample(list(dirty_series.index), sample_size)
        dirty_sample = dirty_series.loc[sample_indices].tolist()
        cleaned_sample = cleaned_series.loc[sample_indices].tolist()
        dirty_sample_fmt = format_list_for_prompt(dirty_sample)
        cleaned_sample_fmt = format_list_for_prompt(cleaned_sample)
        column_comparison_str = "Dirty → Cleaned\n" + "\n".join(
            [f"{d} → {c}" for d, c in zip(dirty_sample_fmt, cleaned_sample_fmt)])
        # 2. Add last attempt message to Validator Agent     
        last_attempt_msg = (
            "\nThis is the final validation attempt. "
            "If the column maintains the same dominant format and style as the original column (no undesired changes such as casing, patterns or structure), approve the column as clean (set 'needs_correction' to False) even if some minor errors remain. "
            "Since partial correction without undesired changes is preferred over reverting to the original column.\n"
        )
        # 3. Instantiate prompt template
        prompt_type = PROMPT_MAP.get(column_type)
        prompt_template = VALIDATION_PROMPT_TEMPLATES.get(prompt_type)
        final_prompt = prompt_template.format(
            last_attempt_msg=last_attempt_msg if last_attempt else "",
            column_name=col,
            column_comparison_sample=column_comparison_str
        )
        return final_prompt

    # ===== Multi-column operations =====
    def create_prompt_recommender_multi_col(self, df: pd.DataFrame, task_info: MultiColumnTask, max_sample_size: int = 40) -> str:
        """Enables multiple multi-column cleaning components to use this function."""
        task_type = task_info.task_type
        if task_type == 'FD':
            return self._create_prompt_recommender_fd(df, task_info, max_sample_size)
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
                    max_sample_size: int = 150) -> str:
        """Enables multiple multi-column cleaning components to use this function."""
        task_type = task_info.task_type
        comparison_str = self._generate_multi_col_comparison(
            dirty_target, cleaned_target, max_sample_size)
        if task_type == 'FD':
            return self._create_prompt_validation_fd(task_info, last_attempt, comparison_str)
        else:
            raise ValueError(f"Unsupported multi-column task type: {task_type}")

    def _generate_multi_col_comparison(self, dirty_target: pd.DataFrame, cleaned_target: pd.DataFrame, max_sample_size: int) -> str:
        """
        Generates comparision string for any number of columns.
        Format: dirty_col1, dirty_col2 => clean_col1, clean_col2
        """
        # 1. Sample dataset
        sample_size = min(len(dirty_target), max_sample_size)
        sample_indices = random.sample(list(dirty_target.index), sample_size)
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
    def _create_prompt_recommender_fd(self, df: pd.DataFrame, task_info: MultiColumnTask, max_sample_size: int) -> str:
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
        fd_pair_sample_str = fd_pair_sample.to_csv(index=False).strip()
        # 3. Instantiate prompt template
        template = FD_RECOMMENDATION_PROMPT_TEMPLATE
        if violation_count > 0:
            header_str = ",".join(df.columns)
            violations = fd_violation_data.get('violations', [])
            violations_str = _format_violations_for_prompt(violations)
            return template.format(
                lhs=lhs,
                rhs=rhs,
                fd_pair_sample=fd_pair_sample_str,
                imputable_count=imputation_count,
                violation_count=violation_count,
                column_header=header_str,
                violations=violations_str)   
        else:
            return template.format(
                lhs=lhs,
                rhs=rhs,
                fd_pair_sample=fd_pair_sample_str,
                imputable_count=imputation_count,
                violation_count=0,
                column_header="-",
                violations="No violations")

    def _create_prompt_coding_fd(self, task_info: MultiColumnTask, recommender_data: dict, allowed_packages: str) -> str:
        fd_data = cast(FDResult, task_info.data)
        return FD_CODING_PROMPT_TEMPLATE.format(
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
        return FD_VALIDATION_PROMPT_TEMPLATE.format(
            lhs=lhs,
            rhs=rhs,
            last_attempt_msg=last_attempt_msg if last_attempt else "",
            fd_comparison_sample=comparison_str)