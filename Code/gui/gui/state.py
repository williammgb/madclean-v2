import reflex as rx
import pandas as pd
import asyncio
import os
import time
import json
import threading
import math
import re
import random
from dataclasses import asdict
from typing import Optional, List, Dict, Any, Callable, Tuple
from madclean.config.settings import CleaningConfig
from madclean.config.loader import load_default_cleaning_config
from madclean.llm.llm_registry import LLM_CLIENT_MAP
from madclean.pipeline import Pipeline
from madclean.components.domain.schema import MultiColumnTask, ColumnProfile, FDResult
from madclean.components.dataprofiler.dataprofiler import DataProfiler
from madclean.components.dataprofiler.outlier_detection import OutlierDetection
from madclean.components.dataprofiler.functional_dependencies import FunctionalDependencies
from madclean.evaluation.dataset_evaluation import (
    check_frames_compatible,
    compute_cleaning_metrics,
    format_pct,
)

_CANCEL_EVENT: threading.Event | None = None
_USER_VALIDATION_CONDITION = threading.Condition()
_PENDING_USER_VALIDATIONS: Dict[str, Dict[str, Any]] = {}
_USER_VALIDATION_RESULTS: Dict[str, Dict[str, Any]] = {}
_HITL_CONDITION = threading.Condition()
_HITL_PENDING: Dict[str, Dict[str, Any]] = {}
_HITL_RESULTS: Dict[str, Dict[str, Any]] = {}

class State(rx.State):
    """Bridge between MADClean system and the UI."""
    _default_config = load_default_cleaning_config()

    # Configuration State
    verbose: bool = _default_config.verbose
    enable_validation: bool = _default_config.enable_validation
    enable_user_validation: bool = _default_config.enable_user_validation
    enable_validation_multi: bool = _default_config.enable_validation_multi
    enable_multi_col_cleaning: bool = _default_config.enable_multi_col_cleaning
    human_in_the_loop: bool = _default_config.human_in_the_loop
    hitl_apply_to_all_columns: bool = _default_config.hitl_apply_to_all_columns
    hitl_selected_columns: List[str] = []
    hitl_column_checkbox_rows: List[Dict[str, str]] = []
    recommender_column_hints: Dict[str, str] = {}
    recommender_column_labeled_examples: Dict[str, str] = {}
    recommender_hint_dialog_open: bool = False
    recommender_hint_editing_column: str = ""
    recommender_hint_editor_text: str = ""
    # User-labeled cells (few-shot for recommender + validator): col -> row_index_str -> {kind, expected}
    label_cells_mode: bool = False
    cell_labels: Dict[str, Dict[str, Dict[str, str]]] = {}
    labeling_dialog_open: bool = False
    labeling_target_col: str = ""
    labeling_target_row_key: str = ""
    labeling_current_value_preview: str = ""
    labeling_kind: str = "clean"
    labeling_expected_value: str = ""
    max_labeled_cells_per_column: int = 10
    llm_temperature_input: str = ""
    llm_top_p_input: str = ""

    # Nested Dictionary for Sample Sizes
    sample_sizes: Dict[str, Dict[str, int]] = json.loads(json.dumps(_default_config.sample_sizes))

    sample_size_validator: int = _default_config.sample_size_validator
    sample_size_validator_random: int = _default_config.sample_size_validator_random
    sample_size_validator_changed: int = _default_config.sample_size_validator_changed
    validator_failure_strategy: str = _default_config.validator_failure_strategy
    max_cleaning_attempts: int = _default_config.max_cleaning_attempts
    max_multi_col_attempts: int = _default_config.max_multi_col_attempts
    max_parse_attempts: int = _default_config.max_parse_attempts
    max_coding_attempts: int = _default_config.max_coding_attempts
    semaphore_limit: int = _default_config.semaphore_limit
    include_metadata: bool = _default_config.include_metadata

    # Execution and progress
    is_cleaning: bool = False
    progress_percent: int = 0
    status_msg: str = "Ready"
    logs: List[str] = []
    last_run_status: str = "idle"  # idle|running|finished|cancelled|error
    _log_file_path: str = ""
    _log_file_offset: int = 0
    _trace_file_path: str = ""
    _trace_file_offset: int = 0
    _max_logs: int = 500
    
    # LLM Settings
    llm_options: List[str] = list(LLM_CLIENT_MAP.keys())
    validation_llm_options: List[str] = llm_options + ["USER"]
    # Per-agent model selection (defaults to the first configured model).
    selected_llm_key_recommender: str = llm_options[0] if llm_options else ""
    selected_llm_key_coding: str = llm_options[0] if llm_options else ""
    selected_llm_key_validation: str = llm_options[0] if llm_options else ""

    # Data State
    file_name: str = "No file selected"
    _file_path: str = ""
    # Optional ground-truth dataset for evaluation (same columns & row count as loaded dirty CSV).
    _gt_df: Optional[pd.DataFrame] = None
    gt_file_name: str = "No ground-truth file"
    evaluation_ready: bool = False
    evaluation_compatible: bool = False
    evaluation_status: str = "Upload a dataset, then optionally upload a matching ground-truth file."
    evaluation_error: str = ""
    evaluation_overall_rows: List[Dict[str, str]] = []
    evaluation_column_rows: List[Dict[str, str]] = []
    df_preview: List[Dict[str, Any]] = []
    df_preview_window: List[Dict[str, Any]] = []
    column_names: List[str] = []
    # Per-column min width for the data table (e.g. "480px"), from longest string on the current page.
    column_min_width: Dict[str, str] = {}
    _full_df: Optional[pd.DataFrame] = None
    _orig_df: Optional[pd.DataFrame] = None
    has_cleaned: bool = False
    modified_cell_keys: List[str] = []  # debug/secondary: f"{row_id}::{col}"
    token_usage: Dict[str, Any] = {}
    token_usage_pretty: str = "{}"
    token_usage_block: str = ""
    token_usage_donut_rows: List[Dict[str, str]] = []
    runtime_seconds: float = 0.0
    runtime_seconds_display: str = "0.0"
    cleaning_summary: Dict[str, Any] = {}
    report_code_keys: List[str] = []
    selected_report_code_key: str = ""
    generated_code_for_selected: str = ""
    generated_code_for_selected_pretty: str = ""
    generated_code_lines_for_selected: List[str] = []
    merged_generated_code: str = ""
    selected_report_meta: Dict[str, Any] = {}
    selected_report_meta_pretty: str = "{}"
    selected_report_meta_lines: List[str] = []
    selected_report_status: str = ""

    column_header_colors: Dict[str, str] = {}  # col -> hex color string
    display_column_header_colors: Dict[str, str] = {}
    user_marked_clean_columns: Dict[str, bool] = {}
    column_profile_rows: List[Dict[str, str]] = []  # [{name, semantic_type, color}]
    profiling_dashboard_rows: List[Dict[str, Any]] = []
    selected_profile_column: str = ""
    regex_pattern_input: str = ""
    regex_results_rows: List[Dict[str, str]] = []
    regex_status: str = "Enter a regex and run detection."

    # Profiling state (types + FDs)
    is_profiling: bool = False
    profiling_status: str = ""
    column_semantic_types: Dict[str, str] = {}  # col -> semantic type
    fd_results: List[Dict[str, Any]] = []  # list of {lhs,rhs,score,violations_count,imputables_count}
    show_fd_graph: bool = False
    show_fds: bool = False
    fd_graph_rows: List[Dict[str, Any]] = []  # grouped as [{"rhs": str, "lhs_values": str, "count": int}]
    selected_fd_keys: List[str] = []
    fd_button_rows: List[Dict[str, str]] = []
    fd_header_badges: Dict[str, List[str]] = {}
    column_fd_markers: Dict[str, str] = {}
    selected_fd_arrows: List[Dict[str, str]] = []
    fd_svg_markup: str = ""

    # Interactive cleaning workflow trace (all columns, live updates).
    column_pipeline_traces: Dict[str, List[Dict[str, str]]] = {}
    pipeline_trace_columns: List[str] = []
    active_trace_step_by_column: Dict[str, str] = {}
    expanded_trace_step_by_column: Dict[str, str] = {}
    pipeline_flow_rows: List[Dict[str, Any]] = []
    pipeline_view_scale: int = 100
    pipeline_card_width: int = 340
    selected_main_tab: str = "table"
    selected_trace_status: str = "Run cleaning to view live column workflows."
    pending_user_validation: bool = False
    pending_validation_request_ids: List[str] = []
    pending_validation_request_id_by_column: Dict[str, str] = {}
    pending_validation_column: str = ""
    pending_validation_attempt: int = 0
    pending_validation_default_target: str = "RECOMMENDER"
    pending_validation_default_feedback: str = ""
    pending_validation_sample_rows: List[Dict[str, str]] = []
    pending_validation_modified_count: int = 0
    pending_validation_width_original: str = "260px"
    pending_validation_width_cleaned: str = "260px"
    user_validation_needs_correction: bool = True
    user_validation_feedback_target: str = "RECOMMENDER"
    user_validation_feedback_message: str = ""
    selected_modified_cell_info: str = ""

    # Human-in-the-loop UI (blocking requests from pipeline thread; multiple columns can be pending at once)
    hitl_last_synced_rid_by_column: Dict[str, str] = {}
    hitl_code_by_column: Dict[str, str] = {}
    hitl_validation_feedback_target_by_column: Dict[str, str] = {}
    hitl_validation_feedback_message_by_column: Dict[str, str] = {}

    # Table pagination (100-row pages).
    page_offset: int = 0
    page_size: int = 100
    total_rows: int = 0
    page_row_count: int = 0
    row_window_size: int = 25
    row_window_offset: int = 0

    # Bottom panel sizing (px).
    bottom_panel_height: int = 240

    # ============================================================
    # EXPLICIT SETTERS
    # ============================================================
    def set_verbose(self, val: bool): self.verbose = val
    def set_enable_validation(self, val: bool): self.enable_validation = val
    def set_enable_user_validation(self, val: bool): self.enable_user_validation = val
    def set_enable_validation_multi(self, val: bool): self.enable_validation_multi = val
    def set_enable_multi_col_cleaning(self, val: bool): self.enable_multi_col_cleaning = val
    def set_include_metadata(self, val: bool): self.include_metadata = val
    def set_selected_llm_key_recommender(self, val: str): self.selected_llm_key_recommender = val
    def set_selected_llm_key_coding(self, val: str): self.selected_llm_key_coding = val
    def set_selected_llm_key_validation(self, val: str):
        self.selected_llm_key_validation = val
        self.enable_user_validation = val == "USER"

    def set_selected_report_code_key(self, val: str):
        self.selected_report_code_key = val
        if val == "ALL_MERGED_CODE":
            self.generated_code_for_selected = self.merged_generated_code
            self.generated_code_for_selected_pretty = self._pretty_python_code(self.generated_code_for_selected)
            self.generated_code_lines_for_selected = self.generated_code_for_selected_pretty.splitlines() or [""]
            self.selected_report_meta = {"datatype": "MERGED", "cleaned": True, "cleaning_validated": True}
            self.selected_report_meta_pretty = json.dumps(self.selected_report_meta, indent=2)
            self.selected_report_meta_lines = [f"{k}: {v}" for k, v in self.selected_report_meta.items()]
            self.selected_report_status = "Merged validated code for one-pass dataset cleaning."
            return
        entry = (self.cleaning_summary or {}).get(val, {})
        if not isinstance(entry, dict):
            self.generated_code_for_selected = ""
            self.generated_code_for_selected_pretty = ""
            self.generated_code_lines_for_selected = []
            self.selected_report_meta = {}
            self.selected_report_meta_pretty = "{}"
            self.selected_report_meta_lines = []
            self.selected_report_status = ""
            return
        self.generated_code_for_selected = str(entry.get("generated_code", "") or "")
        self.generated_code_for_selected_pretty = self._pretty_python_code(self.generated_code_for_selected)
        self.generated_code_lines_for_selected = self.generated_code_for_selected_pretty.splitlines() or [""]
        # Keep only a small subset for readability.
        meta = dict(entry)
        meta.pop("generated_code", None)
        self.selected_report_meta = meta
        # Human-friendly status line.
        if meta.get("already_clean"):
            self.selected_report_status = "Flagged as already clean (no changes applied)."
        elif meta.get("cleaned") and meta.get("cleaning_validated"):
            self.selected_report_status = "Cleaned and validated."
        elif meta.get("cleaned"):
            self.selected_report_status = "Cleaned (validation not enabled or not applicable)."
        elif meta.get("cleaning_validated"):
            self.selected_report_status = "Validated."
        else:
            self.selected_report_status = "Not cleaned / needs review."
        try:
            self.selected_report_meta_pretty = json.dumps(meta, indent=2, default=str)
        except Exception:
            self.selected_report_meta_pretty = str(meta)
        # One key-value per line (more compact than JSON).
        lines: List[str] = []
        for k, v in meta.items():
            if isinstance(v, (dict, list)):
                rendered = json.dumps(v, default=str)
            else:
                rendered = str(v)
            lines.append(f"{k}: {rendered}")
        self.selected_report_meta_lines = lines

    def set_show_fd_graph(self, val: bool):
        self.show_fd_graph = bool(val)

    def set_show_fds(self, val: bool):
        self.show_fds = bool(val)
        if not self.show_fds:
            self.selected_fd_keys = []
            self._rebuild_fd_button_rows()
            self._rebuild_column_fd_markers()
            self._rebuild_selected_fd_arrows()

    def set_selected_main_tab(self, val: str):
        self.selected_main_tab = val

    def _run_evaluation(self) -> None:
        """Refresh evaluation metrics when ground truth and/or cleaned data change."""
        self.evaluation_ready = False
        self.evaluation_compatible = False
        self.evaluation_error = ""
        self.evaluation_overall_rows = []
        self.evaluation_column_rows = []

        if self._orig_df is None or self._orig_df.empty:
            self.evaluation_status = "Load a dirty dataset first."
            return

        if self._gt_df is None or self._gt_df.empty:
            self.evaluation_status = (
                "Upload a ground-truth CSV with the same column names (same order) and row count as the loaded dataset. "
                "Metrics appear after you run cleaning."
            )
            return

        ok, msg = check_frames_compatible(
            self._orig_df, self._gt_df, other_label="Ground truth"
        )
        if not ok:
            self.evaluation_error = msg
            self.evaluation_status = "Ground truth does not match the loaded dataset."
            return

        self.evaluation_compatible = True

        if not self.has_cleaned or self._full_df is None:
            self.evaluation_status = (
                "Ground truth is compatible. Run the pipeline to clean the data; metrics will appear here."
            )
            return

        ok_c, msg_c = check_frames_compatible(
            self._orig_df, self._full_df, other_label="Cleaned output"
        )
        if not ok_c:
            self.evaluation_error = msg_c
            self.evaluation_status = "Cleaned data shape does not match the original dataset."
            return

        metrics = compute_cleaning_metrics(self._orig_df, self._full_df, self._gt_df)
        self.evaluation_ready = True
        self.evaluation_status = ""

        m = metrics
        # Single-row overall metrics table: metrics as columns.
        self.evaluation_overall_rows = [
            {
                "tp": str(m.get("tp", 0)),
                "fp": str(m.get("fp", 0)),
                "tn": str(m.get("tn", 0)),
                "fn": str(m.get("fn", 0)),
                "precision": format_pct(m.get("repair_precision")),
                "recall": format_pct(m.get("repair_recall")),
                "f1": format_pct(m.get("f1_repair")),
            }
        ]

        col_rows: List[Dict[str, str]] = []
        per_col = m.get("per_column") or {}
        for col in self.column_names:
            pc = per_col.get(col) or {}
            col_rows.append(
                {
                    "column": col,
                    "tp": str(pc.get("tp", 0)),
                    "fp": str(pc.get("fp", 0)),
                    "tn": str(pc.get("tn", 0)),
                    "fn": str(pc.get("fn", 0)),
                    "precision": format_pct(pc.get("repair_precision")),
                    "recall": format_pct(pc.get("repair_recall")),
                    "f1": format_pct(pc.get("f1")),
                }
            )
        self.evaluation_column_rows = col_rows

    def _build_token_usage_donut_rows(self, report: Dict[str, Any]) -> None:
        rows: List[Dict[str, str]] = []
        token_usage = (report or {}).get("token_usage", {})
        total_usage = (report or {}).get("total_usage", {}) or {}
        if not isinstance(token_usage, dict):
            self.token_usage_donut_rows = []
            return
        # Grand total across all agents.
        total_tokens = int(total_usage.get("total_tokens", 0) or 0)
        if total_tokens <= 0:
            total_tokens = int(
                sum(
                    int((v or {}).get("total_tokens", 0) or 0)
                    for v in token_usage.values()
                    if isinstance(v, dict)
                )
            )
        if total_tokens <= 0:
            self.token_usage_donut_rows = []
            return

        def _pct_of_total(v: int) -> float:
            return max(0.0, min(100.0, (float(v) / float(total_tokens)) * 100.0))

        # Per-agent donuts: only input vs output segments; center shows share of total tokens.
        ordered_agents = ("recommender", "coding", "validation")
        for agent_key in ordered_agents:
            usage = token_usage.get(agent_key) or {}
            if not isinstance(usage, dict):
                usage = {}
            inp = int(usage.get("input_tokens", 0) or 0)
            out = int(usage.get("output_tokens", 0) or 0)
            agent_total = inp + out
            if agent_total <= 0:
                continue
            # Segment sizes within the donut for this agent.
            inp_frac = max(0.0, min(100.0, (float(inp) / float(agent_total)) * 100.0))
            out_frac = max(
                0.0, min(100.0, (float(out) / float(agent_total)) * 100.0)
            )
            # Center text: agent’s share of overall tokens.
            agent_pct_total = _pct_of_total(agent_total)
            rows.append(
                {
                    "title": agent_key.capitalize(),
                    "subtitle": f"{agent_total:,} tokens",
                    "center_text": f"{agent_pct_total:.1f}%",
                    "legend_items": [
                        {"label": f"Input: {inp:,}", "color": "#2563eb"},
                        {"label": f"Output: {out:,}", "color": "#f97316"},
                    ],
                    "bg": (
                        f"conic-gradient(#2563eb 0% {inp_frac:.2f}%, "
                        f"#f97316 {inp_frac:.2f}% {(inp_frac + out_frac):.2f}%, "
                        f"#e5e7eb {(inp_frac + out_frac):.2f}% 100%)"
                    ),
                }
            )

        # Total donut: share per agent.
        agents_totals: List[tuple[str, int]] = []
        for agent_key in ordered_agents:
            usage = token_usage.get(agent_key) or {}
            if not isinstance(usage, dict):
                continue
            agents_totals.append(
                (agent_key, int(usage.get("total_tokens", 0) or 0))
            )
        # Fallback: any extra agents not in the default order.
        for agent_key, usage in token_usage.items():
            if agent_key in ordered_agents:
                continue
            if not isinstance(usage, dict):
                continue
            agents_totals.append(
                (str(agent_key), int(usage.get("total_tokens", 0) or 0))
            )

        if agents_totals:
            # Build conic gradient segments for each agent.
            segments: List[str] = []
            legend_items: List[Dict[str, str]] = []
            cursor = 0.0
            colors = ["#2563eb", "#f97316", "#22c55e", "#7c3aed", "#ec4899"]
            for idx, (agent_key, atotal) in enumerate(agents_totals):
                if atotal <= 0:
                    continue
                pct = _pct_of_total(atotal)
                start = cursor
                end = min(100.0, cursor + pct)
                color = colors[idx % len(colors)]
                segments.append(f"{color} {start:.2f}% {end:.2f}%")
                legend_items.append(
                    {"label": f"{agent_key.capitalize()}: {atotal:,} ({pct:.1f}%)", "color": color}
                )
                cursor = end
            if segments:
                rows.append(
                    {
                        "title": "Total",
                        "subtitle": f"{total_tokens:,} tokens",
                        "center_text": "100%",
                        "legend_items": legend_items,
                        "bg": f"conic-gradient({', '.join(segments)})",
                    }
                )
        self.token_usage_donut_rows = rows

    def clear_ground_truth(self) -> None:
        self._gt_df = None
        self.gt_file_name = "No ground-truth file"
        if self.status_msg.startswith("Ground truth loaded:"):
            self.status_msg = "Ground truth cleared."
        self._run_evaluation()

    def set_selected_profile_column(self, val: str):
        self.selected_profile_column = val

    def set_regex_pattern_input(self, val: str):
        self.regex_pattern_input = val

    def set_user_validation_needs_correction(self, val: bool):
        self.user_validation_needs_correction = bool(val)

    def set_user_validation_feedback_target(self, val: str):
        if val in ("CODER", "RECOMMENDER"):
            self.user_validation_feedback_target = val

    def set_user_validation_feedback_message(self, val: str):
        self.user_validation_feedback_message = val

    def set_pipeline_view_scale(self, val: List[float]):
        if not val:
            return
        self.pipeline_view_scale = max(60, min(140, int(val[0])))

    def set_pipeline_card_width(self, val: List[float]):
        if not val:
            return
        self.pipeline_card_width = max(240, min(520, int(val[0])))

    def select_pipeline_flow_step(self, column: str, step_id: str):
        expanded = dict(self.expanded_trace_step_by_column or {})
        # Toggle details panel; active step is controlled by live workflow events.
        if expanded.get(column) == step_id:
            expanded[column] = ""
        else:
            expanded[column] = step_id
        self.expanded_trace_step_by_column = expanded
        req_id = (self.pending_validation_request_id_by_column or {}).get(column, "")
        if req_id and req_id in _PENDING_USER_VALIDATIONS:
            pending = _PENDING_USER_VALIDATIONS.get(req_id, {})
            self.pending_validation_column = column
            self.pending_validation_attempt = int(pending.get("attempt", 0) or 0)
            self.pending_validation_default_target = str(pending.get("default_target", "RECOMMENDER"))
            self.pending_validation_default_feedback = str(pending.get("default_feedback", ""))
            self.pending_validation_sample_rows = [
                self._normalize_sample_row(r)
                for r in (pending.get("sample_rows") or [])
                if isinstance(r, dict)
            ]
            self.pending_validation_modified_count = int(pending.get("modified_count", 0) or 0)
            self.user_validation_needs_correction = True
            self.user_validation_feedback_target = self.pending_validation_default_target
            self.user_validation_feedback_message = self.pending_validation_default_feedback
            self._update_pending_validation_widths()
        self._rebuild_pipeline_flow_rows()

    @rx.event
    def stop_pipeline(self):
        """Request cooperative cancellation for the currently running pipeline."""
        global _CANCEL_EVENT
        if _CANCEL_EVENT is not None:
            _CANCEL_EVENT.set()
        # Unblock any callbacks waiting on manual user/HITL responses.
        with _USER_VALIDATION_CONDITION:
            for rid, pending in list(_PENDING_USER_VALIDATIONS.items()):
                _USER_VALIDATION_RESULTS[rid] = {
                    "needs_correction": False,
                    "feedback_target": str(pending.get("default_target", "RECOMMENDER")),
                    "feedback": str(pending.get("default_feedback", "")),
                }
                _PENDING_USER_VALIDATIONS.pop(rid, None)
            _USER_VALIDATION_CONDITION.notify_all()
        with _HITL_CONDITION:
            for rid, pending in list(_HITL_PENDING.items()):
                kind = str(pending.get("kind", "") or "")
                if kind == "code_review":
                    _HITL_RESULTS[rid] = {"code": str(pending.get("code", "") or "")}
                elif kind == "validation_review":
                    _HITL_RESULTS[rid] = {"decision": "feedback_accept"}
                elif kind == "already_clean_review":
                    _HITL_RESULTS[rid] = {"decision": "confirm"}
                else:
                    _HITL_RESULTS[rid] = {}
                _HITL_PENDING.pop(rid, None)
            _HITL_CONDITION.notify_all()
        # UI feedback (actual stop may take a bit while LLM calls finish).
        self.status_msg = "Stopping pipeline..."

    def _update_report_code_view(self, report: Dict[str, Any]):
        # Show all columns/tasks present in the report and add merged code artifact.
        code_keys: list[str] = []
        for k, v in (report or {}).items():
            if not isinstance(v, dict):
                continue
            # Only show per-column / per-task entries, not global report fields.
            if k in ("token_usage", "total_usage"):
                continue
            if k in self.column_names or "→" in k or k.startswith("FD:") or "FD" in k or v.get("target_columns"):
                code_keys.append(k)
        merged_code = self._build_merged_generated_code(report)
        self.merged_generated_code = merged_code
        if merged_code:
            code_keys = ["ALL_MERGED_CODE"] + code_keys
        self.report_code_keys = code_keys
        self.selected_report_code_key = code_keys[0] if code_keys else ""
        if code_keys:
            self.set_selected_report_code_key(code_keys[0])
        else:
            self.generated_code_for_selected = ""
            self.generated_code_for_selected_pretty = ""
            self.generated_code_lines_for_selected = []
            self.selected_report_meta = {}
            self.selected_report_meta_pretty = "{}"
            self.selected_report_meta_lines = []
            self.selected_report_status = ""
            self.selected_report_meta_pretty = "{}"

    @staticmethod
    def _build_merged_generated_code(report: Dict[str, Any]) -> str:
        import re

        import_lines: List[str] = []
        function_blocks: List[str] = []
        apply_lines: List[str] = []

        def _safe_name(key: str) -> str:
            return re.sub(r"[^0-9a-zA-Z_]+", "_", key).strip("_").lower() or "task"

        all_items = [(k, v) for k, v in (report or {}).items() if isinstance(v, dict)]
        ordered_items: List[tuple[str, Dict[str, Any]]] = []
        # Use report order but force single-column keys before FD tasks.
        col_items = [(k, v) for (k, v) in all_items if isinstance(k, str) and v.get("target_columns") is None]
        fd_items = [(k, v) for (k, v) in all_items if v.get("target_columns")]
        ordered_items.extend(col_items)
        ordered_items.extend(fd_items)

        for key, entry in ordered_items:
            if not isinstance(entry, dict):
                continue
            if str(key) == "ALL_MERGED_CODE":
                continue
            # Include only successful cleaned tasks that are not already clean.
            if not bool(entry.get("cleaned")):
                continue
            if bool(entry.get("already_clean")):
                continue
            code = str(entry.get("generated_code", "") or "").strip()
            if not code:
                continue

            lines = code.splitlines()
            local_imports = [ln for ln in lines if ln.strip().startswith("import ") or ln.strip().startswith("from ")]
            local_body_lines = [ln for ln in lines if ln not in local_imports]
            local_body = "\n".join(local_body_lines).strip()
            if not local_body:
                continue
            if "def clean_column" not in local_body:
                continue

            for ln in local_imports:
                if ln not in import_lines:
                    import_lines.append(ln)

            task_fn = f"clean_column_{_safe_name(str(key))}"
            renamed_body = re.sub(r"\bdef\s+clean_column\s*\(", f"def {task_fn}(", local_body, count=1)
            function_blocks.append(f"# --- {key} ---\n{renamed_body}")

            if key in (entry.get("target_columns") or []):
                # Defensive no-op branch; normal column handling is below.
                pass
            if key in (entry.get("target_columns") or []) or entry.get("target_columns"):
                apply_lines.append(f"    df = {task_fn}(df)")
            else:
                col_name = str(key)
                apply_lines.append(f"    df[{col_name!r}] = {task_fn}(df[{col_name!r}])")

        if not function_blocks:
            return ""

        if "import pandas as pd" not in import_lines:
            import_lines.insert(0, "import pandas as pd")

        wrapper_lines = [
            "def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:",
            "    df = df.copy()",
            *apply_lines,
            "    return df",
        ]

        merged = "\n".join(import_lines).strip() + "\n\n" + "\n\n".join(function_blocks) + "\n\n" + "\n".join(wrapper_lines)
        return merged

    # Helper for nested dict updates
    def update_sample_size(self, category: str, subkey: str, value: str):
        if value.isdigit():
            new_sizes = self.sample_sizes.copy()
            new_sizes[category][subkey] = int(value)
            self.sample_sizes = new_sizes
    
    def set_sample_sizes(self, value: list[float]):
        if value:
            new_sizes = self.sample_sizes.copy()
            new_sizes["NUMERIC"]["clean_sample_size"] = int(value[0])
            self.sample_sizes = new_sizes

    def set_sample_size_validator(self, val: str): self.sample_size_validator = int(val) if val.isdigit() else self.sample_size_validator
    def set_sample_size_validator_random(self, val: str): self.sample_size_validator_random = int(val) if val.isdigit() else self.sample_size_validator_random
    def set_sample_size_validator_changed(self, val: str): self.sample_size_validator_changed = int(val) if val.isdigit() else self.sample_size_validator_changed
    def set_validator_failure_strategy(self, val: str):
        if val in ("accept_cleaned", "leave_uncleaned", "ask_user"):
            self.validator_failure_strategy = val
    def set_max_cleaning_attempts(self, val: str): self.max_cleaning_attempts = int(val) if val.isdigit() else self.max_cleaning_attempts
    def set_max_multi_col_attempts(self, val: str): self.max_multi_col_attempts = int(val) if val.isdigit() else self.max_multi_col_attempts
    def set_max_parse_attempts(self, val: str): self.max_parse_attempts = int(val) if val.isdigit() else self.max_parse_attempts
    def set_max_coding_attempts(self, val: str): self.max_coding_attempts = int(val) if val.isdigit() else self.max_coding_attempts
    def set_semaphore_limit(self, val: str): self.semaphore_limit = int(val) if val.isdigit() else self.semaphore_limit
    def set_max_labeled_cells_per_column(self, val: str):
        if val.isdigit():
            self.max_labeled_cells_per_column = max(1, int(val))
            self._sync_labeled_examples_to_config()

    @rx.event
    def prev_page(self):
        self.page_offset = max(0, self.page_offset - self.page_size)
        self.row_window_offset = 0
        self._refresh_current_page()

    @rx.event
    def next_page(self):
        if self.total_rows <= 0:
            return
        max_start = max(0, self.total_rows - self.page_size)
        self.page_offset = min(max_start, self.page_offset + self.page_size)
        self.row_window_offset = 0
        self._refresh_current_page()

    def set_page_from_slider(self, val: List[float]):
        if not val or self.total_rows <= 0:
            return
        # Slider represents 1-based row index; snap to page start.
        row_idx = max(1, min(self.total_rows, int(val[0])))
        self.page_offset = row_idx - 1
        self.row_window_offset = 0
        self._refresh_current_page()

    def set_row_window_offset(self, val: List[float]):
        if not val:
            return
        max_start = max(0, self.page_row_count - self.row_window_size)
        # Slider is 1-based for user readability.
        requested = max(1, int(val[0]))
        self.row_window_offset = max(0, min(max_start, requested - 1))
        self._update_preview_window()

    def toggle_fd_selection(self, lhs: str, rhs: str):
        key = f"{lhs}::{rhs}"
        selected = list(self.selected_fd_keys or [])
        if key in selected:
            selected = []
        else:
            selected = [key]
        self.selected_fd_keys = selected
        self._rebuild_fd_button_rows()
        self._rebuild_column_fd_markers()
        self._rebuild_selected_fd_arrows()

    @rx.event
    def run_regex_detection(self):
        if self._full_df is None or self._full_df.empty:
            self.regex_status = "No data loaded."
            self.regex_results_rows = []
            return
        if not self.selected_profile_column or self.selected_profile_column not in self._full_df.columns:
            self.regex_status = "Select a column first."
            self.regex_results_rows = []
            return
        pattern = (self.regex_pattern_input or "").strip()
        if not pattern:
            self.regex_status = "Enter a regex pattern."
            self.regex_results_rows = []
            return
        try:
            compiled = re.compile(pattern)
        except re.error as ex:
            self.regex_status = f"Invalid regex: {ex}"
            self.regex_results_rows = []
            return
        series = self._full_df[self.selected_profile_column].astype(str).fillna("")
        matches = series[series.apply(lambda v: bool(compiled.search(v)))]
        rows: List[Dict[str, str]] = []
        for idx, val in matches.head(25).items():
            rows.append({"row": str(int(idx) + 1), "value": str(val)})
        self.regex_results_rows = rows
        self.regex_status = (
            f"Pattern matched {int(matches.shape[0])} values in '{self.selected_profile_column}'. Showing up to 25."
        )

    @rx.event
    def submit_user_validation(self, column: str):
        if not self.pending_user_validation:
            return
        request_id = (self.pending_validation_request_id_by_column or {}).get(column, "")
        if not request_id:
            return
        payload = {
            "needs_correction": bool(self.user_validation_needs_correction),
            "feedback_target": self.user_validation_feedback_target,
            "correction_instructions": self.user_validation_feedback_message,
        }
        with _USER_VALIDATION_CONDITION:
            _USER_VALIDATION_RESULTS[request_id] = payload
            _PENDING_USER_VALIDATIONS.pop(request_id, None)
            _USER_VALIDATION_CONDITION.notify_all()
        self.pending_validation_request_id_by_column = {
            k: v for k, v in (self.pending_validation_request_id_by_column or {}).items() if v != request_id
        }
        self.pending_validation_request_ids = [
            rid for rid in (self.pending_validation_request_ids or []) if rid != request_id
        ]
        self.pending_user_validation = len(self.pending_validation_request_ids) > 0
        if self.pending_user_validation:
            next_req_id = self.pending_validation_request_ids[0]
            next_pending = _PENDING_USER_VALIDATIONS.get(next_req_id, {})
            self.pending_validation_column = str(next_pending.get("column", ""))
            self.pending_validation_sample_rows = [
                self._normalize_sample_row(r)
                for r in (next_pending.get("sample_rows") or [])
                if isinstance(r, dict)
            ]
            self.pending_validation_modified_count = int(next_pending.get("modified_count", 0) or 0)
        else:
            self.pending_validation_column = ""
            self.pending_validation_sample_rows = []
            self.pending_validation_modified_count = 0
        self._update_pending_validation_widths()
        self.user_validation_feedback_message = ""
        self.selected_trace_status = "User validation submitted. Pipeline resumed."

    @staticmethod
    def _pretty_python_code(code: str) -> str:
        if not code:
            return ""
        normalized = str(code).replace("\r\n", "\n").replace("\r", "\n").rstrip()
        if not normalized:
            return ""
        # Try to pretty-format using black if available.
        try:
            import black  # type: ignore

            return black.format_str(normalized, mode=black.FileMode())
        except Exception:
            # Keep indentation from generated code for better readability.
            return "\n".join(line.rstrip() for line in normalized.splitlines())

    @rx.event
    def update_cell(self, row_id: int, col: str, value: str):
        """Edit a single cell in the cleaned table (in-place)."""
        if not self.has_cleaned or self._full_df is None:
            return
        if col not in self._full_df.columns:
            return
        try:
            target_dtype = self._full_df[col].dtype
            new_value = self._coerce_value_for_dtype(value, target_dtype)
            self._full_df.at[row_id, col] = new_value
        except Exception:
            return
        self._refresh_current_page()
        if self._gt_df is not None:
            self._run_evaluation()

    # ============================================================
    # HELPERS
    # ============================================================
    @staticmethod
    def _px_for_text_len(max_len: int) -> int:
        """Map max character count to a table column width (px), capped for very long cells."""
        max_len = max(0, min(int(max_len), 100_000))
        min_w, max_w, char = 72, 8000, 7.2
        return int(min(max_w, max(min_w, 16 + max_len * char)))

    @staticmethod
    def _px_for_sample_compare(max_len: int) -> int:
        """Narrower columns for validation / HITL sample tables (easier side-by-side compare)."""
        base = State._px_for_text_len(max_len)
        return int(min(240, max(100, base // 2 + 48)))

    def _compute_column_display_widths(self) -> None:
        """Width per column from header name and longest cell string on the current page."""
        cols = self.column_names
        if not cols:
            self.column_min_width = {}
            return
        widths: Dict[str, str] = {}
        for c in cols:
            max_len = len(str(c))
            for row in self.df_preview or []:
                if c in row:
                    v = row.get(c)
                elif str(c) in row:
                    v = row.get(str(c))
                else:
                    continue
                if v is None:
                    s = ""
                else:
                    try:
                        if isinstance(v, float) and pd.isna(v):
                            s = ""
                        else:
                            s = str(v)
                    except Exception:
                        s = ""
                max_len = max(max_len, len(s))
            widths[str(c)] = f"{self._px_for_text_len(max_len)}px"
        self.column_min_width = widths

    def _update_pending_validation_widths(self) -> None:
        rows = self.pending_validation_sample_rows or []
        o_len = len("Original")
        c_len = len("Cleaned")
        for r in rows:
            o_len = max(o_len, len(str(r.get("original", ""))))
            c_len = max(c_len, len(str(r.get("cleaned", ""))))
        self.pending_validation_width_original = f"{self._px_for_sample_compare(o_len)}px"
        self.pending_validation_width_cleaned = f"{self._px_for_sample_compare(c_len)}px"

    @staticmethod
    def _df_to_preview(df: pd.DataFrame, n: int = 100) -> list[dict[str, Any]]:
        # Add a stable row index for diff-highlighting in the UI.
        flag_cols = list(df.columns)
        head = df.head(n).reset_index(drop=False).rename(columns={"index": "__row_index"})
        records = head.to_dict("records")
        for r in records:
            r["__modified_map"] = {}
            r["__modified_flags"] = [False] * len(flag_cols)
        return records

    def _build_page_preview(self, df: pd.DataFrame, offset: int, n: int) -> list[dict[str, Any]]:
        page = df.iloc[offset : offset + n]
        if page.empty:
            return []
        page = page.reset_index(drop=False).rename(columns={"index": "__row_index"})
        # Include only columns we render (plus row id).
        cols = [c for c in self.column_names if c in page.columns]
        page_records = page[[*cols, "__row_index"]].to_dict("records")
        for i, r in enumerate(page_records):
            # Use both internal and UI-safe keys; dunder keys can be unreliable in event payloads.
            r["__row_id"] = str(offset + i)
            r["row_id"] = str(offset + i)
            r["__modified_flags"] = [False] * len(cols)
            r["__label_flags"] = self._label_flags_full_row(str(r["__row_id"]))
            r["__label_kinds"] = self._label_kinds_full_row(str(r["__row_id"]))
        # Ensure flag list length matches column_names length.
        if len(cols) != len(self.column_names):
            # Map flags to full column order.
            col_to_idx = {c: i for i, c in enumerate(cols)}
            for r in page_records:
                full_flags = [False] * len(self.column_names)
                for c in self.column_names:
                    if c in col_to_idx:
                        full_flags[self.column_names.index(c)] = r["__modified_flags"][col_to_idx[c]]
                r["__modified_flags"] = full_flags
                r["__label_flags"] = self._label_flags_full_row(str(r["__row_id"]))
                r["__label_kinds"] = self._label_kinds_full_row(str(r["__row_id"]))
        return page_records

    def _build_page_preview_with_diff(
        self,
        orig_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        offset: int,
        n: int,
    ) -> list[dict[str, Any]]:
        orig_page = orig_df.iloc[offset : offset + n].reset_index(drop=False).rename(columns={"index": "__row_index"})
        cleaned_page = cleaned_df.iloc[offset : offset + n].reset_index(drop=False).rename(columns={"index": "__row_index"})
        cols = [c for c in self.column_names if c in orig_page.columns and c in cleaned_page.columns]
        if cleaned_page.empty:
            return []

        cleaned_records = cleaned_page[[*cols, "__row_index"]].to_dict("records")
        # Compute modified flags per rendered column order.
        for i in range(min(len(orig_page), len(cleaned_page))):
            modified_flags_for_cols: list[bool] = []
            original_values_for_cols: list[str] = []
            for c in cols:
                a = orig_page.iloc[i][c]
                b = cleaned_page.iloc[i][c]
                modified = not self._safe_cell_equal(a, b)
                modified_flags_for_cols.append(modified)
                original_values_for_cols.append("" if pd.isna(a) else str(a))

            # Write flags back into the matching record by position.
            cleaned_records[i]["__row_id"] = str(offset + i)
            cleaned_records[i]["row_id"] = str(offset + i)
            cleaned_records[i]["__modified_flags"] = modified_flags_for_cols
            cleaned_records[i]["__original_values"] = original_values_for_cols
            cleaned_records[i]["__label_flags"] = self._label_flags_full_row(
                str(cleaned_records[i]["__row_id"])
            )
            cleaned_records[i]["__label_kinds"] = self._label_kinds_full_row(
                str(cleaned_records[i]["__row_id"])
            )

        # Expand flags to full column_names order if needed.
        if len(cols) != len(self.column_names):
            col_to_idx = {c: i for i, c in enumerate(cols)}
            for r in cleaned_records:
                full_flags = [False] * len(self.column_names)
                full_original_values = [""] * len(self.column_names)
                for c_i, c in enumerate(self.column_names):
                    if c in col_to_idx:
                        full_flags[c_i] = r["__modified_flags"][col_to_idx[c]]
                        full_original_values[c_i] = r.get("__original_values", [""] * len(cols))[col_to_idx[c]]
                r["__modified_flags"] = full_flags
                r["__original_values"] = full_original_values
                r["__label_kinds"] = self._label_kinds_full_row(str(r["__row_id"]))
        else:
            # Ensure flags length matches column_names.
            for r in cleaned_records:
                if len(r["__modified_flags"]) != len(self.column_names):
                    r["__modified_flags"] = (r["__modified_flags"] + [False] * len(self.column_names))[: len(self.column_names)]
                if len(r.get("__original_values", [])) != len(self.column_names):
                    r["__original_values"] = (r.get("__original_values", []) + [""] * len(self.column_names))[: len(self.column_names)]
                if len(r.get("__label_kinds", [])) != len(self.column_names):
                    r["__label_kinds"] = self._label_kinds_full_row(str(r["__row_id"]))
        return cleaned_records

    @staticmethod
    def _safe_cell_equal(a: Any, b: Any) -> bool:
        """Safe equality for GUI diff-highlighting.

        Treats numeric-like values as equal even if dtype/format differs (e.g., "3" vs 3.0),
        so columns that become pure numeric after cleaning don't get fully highlighted.
        """
        complex_types = (list, tuple, dict, set)
        try:
            if isinstance(a, complex_types) or isinstance(b, complex_types):
                return str(a) == str(b)
        except Exception:
            pass

        a_na = pd.isna(a)
        b_na = pd.isna(b)
        if bool(a_na) and bool(b_na):
            return True
        if bool(a_na) or bool(b_na):
            return False

        # Numeric-safe comparison: try coercing both to numbers.
        try:
            a_num = pd.to_numeric(a, errors="coerce")
            b_num = pd.to_numeric(b, errors="coerce")
            if not pd.isna(a_num) and not pd.isna(b_num):
                try:
                    return math.isclose(float(a_num), float(b_num), rel_tol=1e-9, abs_tol=0.0)
                except Exception:
                    pass
        except Exception:
            pass

        # Fallback: direct equality, then string equality.
        try:
            return a == b
        except Exception:
            return str(a) == str(b)

    def _refresh_current_page(self):
        if self._full_df is None:
            self.df_preview = []
            self.column_min_width = {}
            return
        if self.total_rows <= 0:
            self.total_rows = int(len(self._full_df))

        offset = max(0, min(self.page_offset, max(0, self.total_rows - 1)))
        # Snap to valid page start.
        page_start = (offset // self.page_size) * self.page_size
        page_start = min(page_start, max(0, self.total_rows - self.page_size))
        self.page_offset = page_start

        if self.has_cleaned and self._orig_df is not None:
            self.df_preview = self._build_page_preview_with_diff(
                self._orig_df, self._full_df, self.page_offset, self.page_size
            )
        else:
            self.df_preview = self._build_page_preview(self._full_df, self.page_offset, self.page_size)
        self.page_row_count = len(self.df_preview)
        self._compute_column_display_widths()
        self._update_preview_window()

    def _update_preview_window(self):
        if not self.df_preview:
            self.df_preview_window = []
            self.page_row_count = 0
            self.row_window_offset = 0
            return
        self.page_row_count = len(self.df_preview)
        max_start = max(0, self.page_row_count - self.row_window_size)
        self.row_window_offset = max(0, min(self.row_window_offset, max_start))
        end = self.row_window_offset + self.row_window_size
        self.df_preview_window = self.df_preview[self.row_window_offset:end]

    def _rebuild_column_fd_markers(self):
        outgoing: Dict[str, List[str]] = {}
        incoming: Dict[str, List[str]] = {}
        for key in self.selected_fd_keys:
            if "::" not in key:
                continue
            lhs, rhs = key.split("::", 1)
            outgoing.setdefault(lhs, [])
            incoming.setdefault(rhs, [])
            if rhs not in outgoing[lhs]:
                outgoing[lhs].append(rhs)
            if lhs not in incoming[rhs]:
                incoming[rhs].append(lhs)
        markers: Dict[str, str] = {}
        for col in self.column_names:
            out_vals = outgoing.get(col, [])
            in_vals = incoming.get(col, [])
            parts: List[str] = []
            if out_vals:
                parts.append("-> " + ", ".join(out_vals[:2]) + ("..." if len(out_vals) > 2 else ""))
            if in_vals:
                parts.append("<- " + ", ".join(in_vals[:2]) + ("..." if len(in_vals) > 2 else ""))
            markers[col] = " | ".join(parts)
        self.column_fd_markers = markers

    def _rebuild_selected_fd_arrows(self):
        total_cols = max(1, len(self.column_names))
        col_idx = {c: i for i, c in enumerate(self.column_names)}
        arrows: List[Dict[str, str]] = []
        lane_count = 6
        ranked: List[tuple[int, int, str]] = []
        for key in self.selected_fd_keys:
            if "::" not in key:
                continue
            lhs, rhs = key.split("::", 1)
            if lhs not in col_idx or rhs not in col_idx:
                continue
            span = abs(col_idx[lhs] - col_idx[rhs])
            ranked.append((span, min(col_idx[lhs], col_idx[rhs]), key))
        ranked.sort(key=lambda t: (-t[0], t[1], t[2]))
        for i, (_span, _order_idx, key) in enumerate(ranked):
            if "::" not in key:
                continue
            lhs, rhs = key.split("::", 1)
            if lhs not in col_idx or rhs not in col_idx:
                continue
            i_lhs = col_idx[lhs]
            i_rhs = col_idx[rhs]
            if i_lhs == i_rhs:
                continue
            from_i = min(i_lhs, i_rhs)
            to_i = max(i_lhs, i_rhs)
            start_pct = ((from_i + 0.5) / total_cols) * 100.0
            end_pct = ((to_i + 0.5) / total_cols) * 100.0
            width_pct = max(1.0, end_pct - start_pct)
            lane = i % lane_count
            top_px = 6 + lane * 10
            direction = "right" if i_lhs < i_rhs else "left"
            lhs_x = start_pct if i_lhs < i_rhs else end_pct
            rhs_x = end_pct if i_lhs < i_rhs else start_pct
            arrows.append(
                {
                    "key": key,
                    "left": f"{start_pct:.4f}%",
                    "width": f"{width_pct:.4f}%",
                    "top": f"{top_px}px",
                    "direction": direction,
                    "lhs_x": f"{lhs_x:.4f}%",
                    "rhs_x": f"{rhs_x:.4f}%",
                }
            )
        self.selected_fd_arrows = arrows
        self._rebuild_fd_svg_markup()

    def _rebuild_fd_svg_markup(self):
        # Build an SVG string with angular orthogonal FD paths.
        width = 1200
        height = 84
        cols = max(1, len(self.column_names))
        col_x = {c: ((i + 0.5) / cols) * width for i, c in enumerate(self.column_names)}
        lane_count = 8

        ranked: List[tuple[int, int, str, str]] = []
        for key in self.selected_fd_keys:
            if "::" not in key:
                continue
            lhs, rhs = key.split("::", 1)
            if lhs not in col_x or rhs not in col_x or lhs == rhs:
                continue
            span = abs(self.column_names.index(lhs) - self.column_names.index(rhs))
            ranked.append((span, min(self.column_names.index(lhs), self.column_names.index(rhs)), lhs, rhs))
        ranked.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))

        paths: List[str] = []
        for i, (_span, _ord, lhs, rhs) in enumerate(ranked):
            x1 = col_x[lhs]
            x2 = col_x[rhs]
            lane = i % lane_count
            y_ctrl = 10 + lane * 7
            # Orthogonal path: up, horizontal, down.
            d = f"M {x1:.2f} 74 L {x1:.2f} {y_ctrl:.2f} L {x2:.2f} {y_ctrl:.2f} L {x2:.2f} 66"
            paths.append(
                f'<path d="{d}" stroke="#111827" stroke-width="2" fill="none" '
                f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#fdArrowHead)" opacity="0.95"/>'
            )
            paths.append(f'<circle cx="{x1:.2f}" cy="74" r="2.6" fill="#111827" opacity="0.95"/>')

        if not paths:
            self.fd_svg_markup = '<svg width="100%" height="84" viewBox="0 0 1200 84" preserveAspectRatio="none"></svg>'
            return

        svg = (
            '<svg width="100%" height="84" viewBox="0 0 1200 84" preserveAspectRatio="none" '
            'xmlns="http://www.w3.org/2000/svg">'
            '<defs>'
            '<marker id="fdArrowHead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
            '<path d="M 0 0 L 8 4 L 0 8 z" fill="#111827"/>'
            "</marker>"
            "</defs>"
            + "".join(paths)
            + "</svg>"
        )
        self.fd_svg_markup = svg

    def _rebuild_fd_button_rows(self):
        selected = set(self.selected_fd_keys or [])
        rows: List[Dict[str, str]] = []
        for fd in self.fd_results:
            lhs = str(fd.get("lhs", "") or "")
            rhs = str(fd.get("rhs", "") or "")
            if not lhs or not rhs:
                continue
            key = f"{lhs}::{rhs}"
            rows.append(
                {
                    "lhs": lhs,
                    "rhs": rhs,
                    "active": "1" if key in selected else "0",
                }
            )
        self.fd_button_rows = rows

    def _rebuild_fd_header_badges(self):
        grouped: Dict[str, List[str]] = {str(c): [] for c in (self.column_names or [])}
        for fd in self.fd_results or []:
            lhs = str(fd.get("lhs", "") or "").strip()
            rhs = str(fd.get("rhs", "") or "").strip()
            if not lhs or not rhs:
                continue
            label = f"{lhs} -> {rhs}"
            grouped.setdefault(lhs, [])
            if label not in grouped[lhs]:
                grouped[lhs].append(label)
        self.fd_header_badges = grouped

    @staticmethod
    def _semantic_type_to_color(semantic_type: str) -> str:
        # Distinct semantic color palette for visual readability.
        mapping = {
            "INTEGER": "#00509d",
            "FLOAT": "#00b4d8",
            "DIRTY_INTEGER": "#4d779e",
            "DIRTY_FLOAT": "#8ac5d1",
            "DIRTY_NUMERIC": "#a0afb7",
            "DISCRETE_STRING": "#7209b7",
            "DELIMITED_STRING": "#ebe70c",
            "COLLECTION": "#afeb0c",
            "NAMED_ENTITY": "#b5179e",
            "NATURAL_LANGUAGE_TEXT": "#ff85a1",
            "BOOLEAN": "#fb8500",
            "DATETIME": "#2d6a4f",
            "STRING": "#7209b7",
            "EMPTY": "#9CA3AF",  # gray
            "UNKNOWN": "#9CA3AF",
        }
        return mapping.get(semantic_type.upper(), "#9CA3AF")

    @staticmethod
    def _coerce_value_for_dtype(value: str, dtype: Any) -> Any:
        """Coerce text input to a dtype-compatible scalar for safe DataFrame assignment."""
        text = value.strip() if isinstance(value, str) else value
        if isinstance(text, str) and text == "":
            if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_datetime64_any_dtype(dtype):
                return pd.NA
            return ""
        try:
            if pd.api.types.is_integer_dtype(dtype):
                numeric = pd.to_numeric(text, errors="coerce")
                return pd.NA if pd.isna(numeric) else int(float(numeric))
            if pd.api.types.is_float_dtype(dtype):
                numeric = pd.to_numeric(text, errors="coerce")
                return pd.NA if pd.isna(numeric) else float(numeric)
            if pd.api.types.is_bool_dtype(dtype):
                if isinstance(text, str):
                    lowered = text.lower()
                    if lowered in ("true", "1", "yes", "y", "t"):
                        return True
                    if lowered in ("false", "0", "no", "n", "f"):
                        return False
                return bool(text)
            if pd.api.types.is_datetime64_any_dtype(dtype):
                parsed = pd.to_datetime(text, errors="coerce")
                return pd.NA if pd.isna(parsed) else parsed
        except Exception:
            return value
        return value

    @staticmethod
    def _trace_output_to_text(payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        try:
            return json.dumps(payload, indent=2, default=str)
        except Exception:
            return str(payload)

    def _build_column_pipeline_trace(self, column_name: str, entry: Dict[str, Any]) -> List[Dict[str, str]]:
        if not isinstance(entry, dict):
            return []
        raw_steps = entry.get("trace_steps", [])
        steps: List[Dict[str, str]] = []
        if isinstance(raw_steps, list):
            for i, s in enumerate(raw_steps):
                if not isinstance(s, dict):
                    continue
                step_id = str(s.get("id", f"step_{i+1}"))
                steps.append(
                    {
                        "id": step_id,
                        "title": str(s.get("title", step_id.replace("_", " ").title())),
                        "status": str(s.get("status", "")),
                        "output": self._trace_output_to_text(s.get("output")),
                    }
                )
        if not steps:
            steps = [
                {
                    "id": "summary",
                    "title": "Result",
                    "status": "completed" if bool(entry.get("cleaned")) else "not_cleaned",
                    "output": self._trace_output_to_text(entry),
                }
            ]
        steps.append(
            {
                "id": "finished",
                "title": "Finished",
                "status": "completed" if (bool(entry.get("cleaned")) or bool(entry.get("already_clean"))) else "failed",
                "output": (
                    "Column: "
                    + str(column_name)
                    + "\nDatatype: "
                    + str(entry.get("datatype", ""))
                    + "\nCleaned: "
                    + str(bool(entry.get("cleaned", False)))
                    + "\nValidated: "
                    + str(bool(entry.get("cleaning_validated", False)))
                    + "\nAttempts: "
                    + str(entry.get("attempts", 0))
                    + "\nAlready clean: "
                    + str(bool(entry.get("already_clean", False)))
                ),
            }
        )
        return steps

    def _update_pipeline_trace_view(self):
        trace_map: Dict[str, List[Dict[str, str]]] = {}
        report = self.cleaning_summary or {}
        for key, entry in report.items():
            if key in ("token_usage", "total_usage", "runtime_seconds", "cancelled", "ALL_MERGED_CODE"):
                continue
            trace_map[key] = self._build_column_pipeline_trace(key, entry if isinstance(entry, dict) else {})
        for col, is_clean in (self.user_marked_clean_columns or {}).items():
            if not is_clean:
                continue
            trace_map[col] = [
                {
                    "id": "user_marked_clean",
                    "title": "User marked clean",
                    "status": "already_clean",
                    "output": "User determined this column is already clean. Pipeline skipped automatic cleaning for this column.",
                },
                {
                    "id": "finished",
                    "title": "Finished",
                    "status": "completed",
                    "output": "Column marked already clean by user.",
                },
            ]
        self.column_pipeline_traces = trace_map
        normal_cols = [c for c in self.column_names if c in trace_map]
        extra_tasks = [k for k in trace_map.keys() if k not in normal_cols]
        self.pipeline_trace_columns = normal_cols + extra_tasks
        active = dict(self.active_trace_step_by_column or {})
        expanded = dict(self.expanded_trace_step_by_column or {})
        for col in self.pipeline_trace_columns:
            steps = self.column_pipeline_traces.get(col, [])
            if steps and col not in active:
                active[col] = steps[-1].get("id", "")
            if col not in expanded:
                expanded[col] = ""
        self.active_trace_step_by_column = active
        self.expanded_trace_step_by_column = expanded
        self._rebuild_pipeline_flow_rows()

    def _hitl_widths_for_sample_rows(self, rows: List[Dict[str, str]]) -> tuple[str, str]:
        o_len = len("Original")
        c_len = len("Cleaned")
        for r in rows or []:
            o_len = max(o_len, len(str(r.get("original", ""))))
            c_len = max(c_len, len(str(r.get("cleaned", ""))))
        return (
            f"{self._px_for_sample_compare(o_len)}px",
            f"{self._px_for_sample_compare(c_len)}px",
        )

    def _hitl_attachments_for_column(self, col: str, pending_by_col: Dict[str, Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        empty: Dict[str, Any] = {
            "hitl_request_id": "",
            "hitl_kind": "",
            "hitl_code": "",
            "hitl_validator_summary_lines": [],
            "hitl_sample_rows": [],
            "hitl_modified_count": "0",
            "hitl_llm_needs_correction": "0",
            "hitl_feedback_target": "RECOMMENDER",
            "hitl_feedback_message": "",
            "hitl_width_original": "260px",
            "hitl_width_cleaned": "260px",
        }
        if col not in pending_by_col:
            return empty
        rid, p = pending_by_col[col]
        kind = str(p.get("kind", "") or "")
        out = dict(empty)
        out["hitl_request_id"] = rid
        out["hitl_kind"] = kind
        if kind == "code_review":
            out["hitl_code"] = (self.hitl_code_by_column or {}).get(col, str(p.get("code", "") or ""))
            return out
        if kind == "validation_review":
            nc = bool(p.get("llm_needs_correction", False))
            out["hitl_llm_needs_correction"] = "1" if nc else "0"
            ft = str(p.get("llm_feedback_target") or "")
            ci = str(p.get("llm_correction_instructions") or "")
            if not nc:
                lines = ["The validation agent determined that the cleaning operations are valid."]
            else:
                lines = [
                    "The validation agent detected possible issues with the cleaning (needs_correction = true)."
                ]
                if ft:
                    lines.append(f"Suggested feedback target: {ft}.")
                if ci.strip():
                    lines.append(f"Correction instructions: {ci.strip()}")
            out["hitl_validator_summary_lines"] = lines
            sample_rows = [
                self._normalize_sample_row(r) for r in (p.get("sample_rows") or []) if isinstance(r, dict)
            ]
            out["hitl_sample_rows"] = sample_rows
            out["hitl_modified_count"] = str(int(p.get("modified_count", 0) or 0))
            wo, wc = self._hitl_widths_for_sample_rows(sample_rows)
            out["hitl_width_original"] = wo
            out["hitl_width_cleaned"] = wc
            uft = (self.hitl_validation_feedback_target_by_column or {}).get(col, ft or "RECOMMENDER")
            if uft not in ("CODER", "RECOMMENDER"):
                uft = "RECOMMENDER"
            out["hitl_feedback_target"] = uft
            out["hitl_feedback_message"] = (self.hitl_validation_feedback_message_by_column or {}).get(col, "")
            return out
        if kind == "already_clean_review":
            out["hitl_validator_summary_lines"] = [
                "The recommender marked this column as already clean. Confirm after sampling values, or reject and describe what is wrong.",
            ]
            sample_rows = [
                self._normalize_sample_row(r) for r in (p.get("sample_rows") or []) if isinstance(r, dict)
            ]
            out["hitl_sample_rows"] = sample_rows
            wo, wc = self._hitl_widths_for_sample_rows(sample_rows)
            out["hitl_width_original"] = wo
            out["hitl_width_cleaned"] = wc
            out["hitl_feedback_message"] = (self.hitl_validation_feedback_message_by_column or {}).get(col, "")
            return out
        return empty

    def _sync_hitl_editing_state_from_global_pending(self) -> Dict[str, Tuple[str, Dict[str, Any]]]:
        with _HITL_CONDITION:
            pending_by_col: Dict[str, Tuple[str, Dict[str, Any]]] = {}
            for rid, p in _HITL_PENDING.items():
                c = str(p.get("column", "") or "").strip()
                if c:
                    pending_by_col[c] = (rid, dict(p))

        last_sync = dict(self.hitl_last_synced_rid_by_column or {})
        code_map = dict(self.hitl_code_by_column or {})
        ft_map = dict(self.hitl_validation_feedback_target_by_column or {})
        fm_map = dict(self.hitl_validation_feedback_message_by_column or {})

        active = set(pending_by_col.keys())
        for col in list(last_sync.keys()):
            if col not in active:
                last_sync.pop(col, None)
                code_map.pop(col, None)
                ft_map.pop(col, None)
                fm_map.pop(col, None)
        for col in list(code_map.keys()):
            if col not in active:
                code_map.pop(col, None)
        for col in list(ft_map.keys()):
            if col not in active:
                ft_map.pop(col, None)
        for col in list(fm_map.keys()):
            if col not in active:
                fm_map.pop(col, None)

        for col, (rid, p) in pending_by_col.items():
            prev_rid = last_sync.get(col, "")
            if rid == prev_rid:
                continue
            last_sync[col] = rid
            kind = str(p.get("kind", "") or "")
            if kind == "code_review":
                code_map[col] = str(p.get("code", "") or "")
            elif kind == "validation_review":
                ft = str(p.get("llm_feedback_target") or "RECOMMENDER")
                if ft not in ("CODER", "RECOMMENDER"):
                    ft = "RECOMMENDER"
                ft_map[col] = ft
                fm_map[col] = ""
            elif kind == "already_clean_review":
                fm_map[col] = ""

        self.hitl_last_synced_rid_by_column = last_sync
        self.hitl_code_by_column = code_map
        self.hitl_validation_feedback_target_by_column = ft_map
        self.hitl_validation_feedback_message_by_column = fm_map
        return pending_by_col

    def _rebuild_pipeline_flow_rows(self):
        pending_by_col = self._sync_hitl_editing_state_from_global_pending()
        rows: List[Dict[str, Any]] = []
        normal_cols = [c for c in self.column_names if c in self.column_pipeline_traces]
        extra_tasks = [k for k in self.column_pipeline_traces.keys() if k not in normal_cols]
        ordered_cols = normal_cols + extra_tasks
        active_map = self.active_trace_step_by_column or {}
        expanded_map = self.expanded_trace_step_by_column or {}
        for col in ordered_cols:
            hitl_pair = pending_by_col.get(col)
            hitl_kind = str(hitl_pair[1].get("kind", "") or "") if hitl_pair else ""
            steps = self.column_pipeline_traces.get(col, [])
            active_id = active_map.get(col, steps[-1].get("id", "") if steps else "")
            expanded_id = expanded_map.get(col, "")
            detail_title = ""
            detail_status = ""
            detail_output_lines: List[str] = []
            column_done = "0"
            rendered_steps: List[Dict[str, str]] = []
            step_lookup: Dict[str, Dict[str, str]] = {}
            for step in steps:
                sid = str(step.get("id", "") or "")
                is_active = sid == active_id
                is_expanded = sid == expanded_id and expanded_id != ""
                status = str(step.get("status", ""))
                if (sid == "finished" and status == "completed") or status == "already_clean":
                    column_done = "1"
                step_lookup[sid] = {
                    "title": str(step.get("title", sid)),
                    "status": status,
                    "output": str(step.get("output", "") or ""),
                }
                needs_blink = "0"
                if status == "needs_user_validation":
                    needs_blink = "1"
                elif hitl_pair:
                    if hitl_kind == "code_review" and sid.startswith("coder_"):
                        needs_blink = "1"
                    elif hitl_kind == "validation_review" and sid.startswith("validator_") and sid != "validator_already_clean":
                        needs_blink = "1"
                    elif hitl_kind == "already_clean_review" and sid == "validator_already_clean":
                        needs_blink = "1"
                if status in ("pending_hitl",):
                    needs_blink = "1"
                rendered_steps.append(
                    {
                        "id": sid,
                        "title": step_lookup[sid]["title"],
                        "status": status,
                        "is_error": "1"
                        if status
                        in ("needs_correction", "failed", "invalid_response", "needs_user_validation", "rejected", "pending_hitl")
                        else "0",
                        "is_success": "1" if status in ("ok", "approved", "completed", "already_clean") else "0",
                        "is_active": "1" if is_active else "0",
                        "is_expanded": "1" if is_expanded else "0",
                        "needs_blink": needs_blink,
                        "show_arrow_after": "0",
                    }
                )
            for i in range(max(0, len(rendered_steps) - 1)):
                rendered_steps[i]["show_arrow_after"] = "1"
            detail_id = expanded_id if expanded_id else active_id
            if detail_id and detail_id in step_lookup:
                detail_title = step_lookup[detail_id]["title"]
                detail_status = step_lookup[detail_id]["status"]
                output = step_lookup[detail_id]["output"]
                detail_output_lines = output.splitlines() if output else [""]
            ex = str(expanded_id or "")
            hitl_ex = self._hitl_attachments_for_column(col, pending_by_col)
            row_out: Dict[str, Any] = {
                "column": col,
                "steps": rendered_steps,
                "active_title": detail_title,
                "active_status": detail_status,
                "active_output_lines": detail_output_lines,
                "show_details": "1" if expanded_id else "0",
                "expanded_step_id": expanded_id,
                "expanded_is_coder_step": "1" if ex.startswith("coder_") else "0",
                "expanded_is_validator_step": "1"
                if (ex.startswith("validator_") and ex != "validator_already_clean")
                else "0",
                "expanded_is_already_clean_step": "1" if ex == "validator_already_clean" else "0",
                "column_done": column_done,
            }
            row_out.update(hitl_ex)
            rows.append(row_out)
        self.pipeline_flow_rows = rows
        if rows:
            self.selected_trace_status = f"Showing live workflow for {len(rows)} columns."
        else:
            self.selected_trace_status = "Run cleaning to view live column workflows."

    def _rebuild_fd_graph_rows(self):
        grouped: Dict[str, List[str]] = {}
        for fd in self.fd_results:
            rhs = str(fd.get("rhs", "") or "")
            lhs = str(fd.get("lhs", "") or "")
            if not rhs:
                continue
            grouped.setdefault(rhs, [])
            if lhs and lhs not in grouped[rhs]:
                grouped[rhs].append(lhs)
        rows: List[Dict[str, Any]] = []
        for rhs, lhs_list in grouped.items():
            rows.append(
                {
                    "rhs": rhs,
                    "lhs_values": ", ".join(lhs_list) if lhs_list else "(none)",
                    "count": len(lhs_list),
                }
            )
        rows.sort(key=lambda r: r["rhs"])
        self.fd_graph_rows = rows

    def _df_to_preview_with_modified_map(
        self,
        dirty_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        n: int = 100,
    ) -> list[dict[str, Any]]:
        dirty_head = dirty_df.head(n).reset_index(drop=False).rename(columns={"index": "__row_index"})
        cleaned_head = cleaned_df.head(n).reset_index(drop=False).rename(columns={"index": "__row_index"})
        common_cols = [c for c in dirty_head.columns if c in cleaned_head.columns and c != "__row_index"]

        for r_idx in range(min(len(dirty_head), len(cleaned_head))):
            dirty_row = dirty_head.iloc[r_idx]
            cleaned_row = cleaned_head.iloc[r_idx]
            row_id = cleaned_row["__row_index"]
            modified_flags: list[bool] = []
            for col in common_cols:
                a = dirty_row[col]
                b = cleaned_row[col]
                modified = not ((pd.isna(a) and pd.isna(b)) or a == b)
                modified_flags.append(modified)
                if modified:
                    self.modified_cell_keys.append(f"{row_id}::{col}")
            cleaned_head.at[r_idx, "__modified_map"] = {c: f for c, f in zip(common_cols, modified_flags)}
            cleaned_head.at[r_idx, "__modified_flags"] = modified_flags

        # Ensure field exists on all records
        records = cleaned_head.to_dict("records")
        for r in records:
            r.setdefault("__modified_map", {})
            r.setdefault("__modified_flags", [False] * len(common_cols))
        return records

    def _enqueue_log(self, msg: str):
        if not msg:
            return
        try:
            if not self._log_file_path:
                return
            with open(self._log_file_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            # Best-effort: never crash the pipeline due to UI logging.
            return

    async def _drain_logs(self):
        if not self._log_file_path or not os.path.exists(self._log_file_path):
            return
        try:
            with open(self._log_file_path, "r", encoding="utf-8") as f:
                f.seek(self._log_file_offset)
                chunk = f.read()
                new_offset = f.tell()
        except Exception:
            return
        if not chunk:
            return
        new_lines = [line for line in chunk.splitlines() if line.strip()]
        if not new_lines:
            return
        async with self:
            self.logs = (self.logs + new_lines)[-self._max_logs :]
            self._log_file_offset = new_offset

    def _enqueue_trace_event(self, event):
        if not event:
            return
        try:
            if not self._trace_file_path:
                return
            with open(self._trace_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event), default=str) + "\n")
        except Exception:
            return

    def _apply_trace_event(self, event: Dict[str, Any]):
        col = str(event.get("column", "") or "")
        if not col:
            return
        step_id = str(event.get("step_id", "") or "")
        if not step_id:
            return
        title = str(event.get("title", step_id) or step_id)
        status = str(event.get("status", "") or "")
        output = str(event.get("output", "") or "")

        trace_map = dict(self.column_pipeline_traces or {})
        steps = list(trace_map.get(col, []))
        replaced = False
        for i, step in enumerate(steps):
            if step.get("id") == step_id:
                steps[i] = {"id": step_id, "title": title, "status": status, "output": output}
                replaced = True
                break
        if not replaced:
            steps.append({"id": step_id, "title": title, "status": status, "output": output})
        trace_map[col] = steps
        self.column_pipeline_traces = trace_map
        self.pipeline_trace_columns = [c for c in self.column_names if c in self.column_pipeline_traces]

        active = dict(self.active_trace_step_by_column or {})
        active[col] = step_id
        self.active_trace_step_by_column = active
        self._rebuild_pipeline_flow_rows()

    async def _drain_trace_events(self):
        if not self._trace_file_path or not os.path.exists(self._trace_file_path):
            async with self:
                self._rebuild_pipeline_flow_rows()
            return
        try:
            with open(self._trace_file_path, "r", encoding="utf-8") as f:
                f.seek(self._trace_file_offset)
                chunk = f.read()
                new_offset = f.tell()
        except Exception:
            return
        if not chunk:
            async with self:
                self._rebuild_pipeline_flow_rows()
            return
        raw_lines = [line for line in chunk.splitlines() if line.strip()]
        events: List[Dict[str, Any]] = []
        for line in raw_lines:
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    events.append(parsed)
            except Exception:
                continue
        if not events:
            async with self:
                pending_map = dict(_PENDING_USER_VALIDATIONS or {})
                if pending_map:
                    self.pending_user_validation = True
                    self.pending_validation_request_ids = list(pending_map.keys())
                    self.pending_validation_request_id_by_column = {
                        str(v.get("column", "")): k for k, v in pending_map.items() if isinstance(v, dict)
                    }
                    # Keep currently selected column if still pending; otherwise switch to first pending.
                    active_col = self.pending_validation_column
                    if active_col not in self.pending_validation_request_id_by_column:
                        active_col = str(next(iter(self.pending_validation_request_id_by_column.keys()), ""))
                    req_id = self.pending_validation_request_id_by_column.get(active_col, "")
                    pending = pending_map.get(req_id, {})
                    self.pending_validation_column = active_col
                    self.pending_validation_attempt = int(pending.get("attempt", 0) or 0)
                    self.pending_validation_default_target = str(pending.get("default_target", "RECOMMENDER"))
                    self.pending_validation_default_feedback = str(pending.get("default_feedback", ""))
                    self.pending_validation_sample_rows = [
                        self._normalize_sample_row(r)
                        for r in (pending.get("sample_rows") or [])
                        if isinstance(r, dict)
                    ]
                    self.pending_validation_modified_count = int(pending.get("modified_count", 0) or 0)
                    self.user_validation_needs_correction = True
                    self.user_validation_feedback_target = self.pending_validation_default_target
                    self.user_validation_feedback_message = self.pending_validation_default_feedback
                    self._update_pending_validation_widths()
                    self.selected_trace_status = "Validation requires your decision. Open the blinking red validator step."
                else:
                    self.pending_user_validation = False
                    self.pending_validation_request_ids = []
                    self.pending_validation_request_id_by_column = {}
                    self.pending_validation_modified_count = 0
                self._rebuild_pipeline_flow_rows()
            return
        async with self:
            for event in events:
                self._apply_trace_event(event)
            self._trace_file_offset = new_offset
            pending_map = dict(_PENDING_USER_VALIDATIONS or {})
            if pending_map:
                self.pending_user_validation = True
                self.pending_validation_request_ids = list(pending_map.keys())
                self.pending_validation_request_id_by_column = {
                    str(v.get("column", "")): k for k, v in pending_map.items() if isinstance(v, dict)
                }
                active_col = self.pending_validation_column
                if active_col not in self.pending_validation_request_id_by_column:
                    active_col = str(next(iter(self.pending_validation_request_id_by_column.keys()), ""))
                req_id = self.pending_validation_request_id_by_column.get(active_col, "")
                pending = pending_map.get(req_id, {})
                self.pending_validation_column = active_col
                self.pending_validation_attempt = int(pending.get("attempt", 0) or 0)
                self.pending_validation_default_target = str(pending.get("default_target", "RECOMMENDER"))
                self.pending_validation_default_feedback = str(pending.get("default_feedback", ""))
                self.pending_validation_sample_rows = [
                    self._normalize_sample_row(r)
                    for r in (pending.get("sample_rows") or [])
                    if isinstance(r, dict)
                ]
                self.pending_validation_modified_count = int(pending.get("modified_count", 0) or 0)
                self.user_validation_needs_correction = True
                self.user_validation_feedback_target = self.pending_validation_default_target
                self.user_validation_feedback_message = self.pending_validation_default_feedback
                self._update_pending_validation_widths()
                self.selected_trace_status = "Validation requires your decision. Open the blinking red validator step."
            else:
                self.pending_user_validation = False
                self.pending_validation_request_ids = []
                self.pending_validation_request_id_by_column = {}
                self.pending_validation_modified_count = 0
            self._rebuild_pipeline_flow_rows()

    def _compute_modified_cell_keys(self, dirty_df: pd.DataFrame, cleaned_df: pd.DataFrame, n: int = 100) -> list[str]:
        dirty_head = dirty_df.head(n).reset_index(drop=True)
        cleaned_head = cleaned_df.head(n).reset_index(drop=True)
        common_cols = [c for c in dirty_head.columns if c in cleaned_head.columns]
        keys: list[str] = []
        for r in range(min(len(dirty_head), len(cleaned_head))):
            for c in common_cols:
                a = dirty_head.at[r, c]
                b = cleaned_head.at[r, c]
                # Treat NaN/None as equal.
                if (pd.isna(a) and pd.isna(b)) or a == b:
                    continue
                keys.append(f"{r}::{c}")
        return keys

    def _build_profiling_dashboard_rows(self):
        if self._full_df is None or self._full_df.empty:
            self.profiling_dashboard_rows = []
            return
        def _regex_signature(value: str) -> str:
            tokens: List[str] = []
            for ch in value:
                if ch.isdigit():
                    tok = r"\d"
                elif ch.isalpha() and ch.isupper():
                    tok = r"[A-Z]"
                elif ch.isalpha() and ch.islower():
                    tok = r"[a-z]"
                elif ch.isspace():
                    tok = r"\s"
                else:
                    tok = re.escape(ch)
                tokens.append(tok)
            if not tokens:
                return r"^$"
            compressed: List[str] = []
            run_tok = tokens[0]
            run_count = 1
            for tok in tokens[1:]:
                if tok == run_tok:
                    run_count += 1
                else:
                    compressed.append(f"{run_tok}{{{run_count}}}" if run_count > 1 else run_tok)
                    run_tok = tok
                    run_count = 1
            compressed.append(f"{run_tok}{{{run_count}}}" if run_count > 1 else run_tok)
            return "^" + "".join(compressed) + "$"

        rows: List[Dict[str, Any]] = []
        n_rows = max(1, int(len(self._full_df)))
        for col in self.column_names:
            if col not in self._full_df.columns:
                continue
            series = self._full_df[col]
            non_null = series.dropna()
            missing_count = int(series.isna().sum())
            unique_count = int(non_null.nunique(dropna=True))
            uniqueness_ratio = float(unique_count / max(1, len(non_null)))
            semantic_type = self.column_semantic_types.get(col, "UNKNOWN")
            is_numeric = bool(pd.api.types.is_numeric_dtype(series))
            numeric_min = ""
            numeric_max = ""
            numeric_std = ""
            numeric_mean = ""
            numeric_median = ""
            numeric_range = ""

            value_counts = non_null.astype(str).value_counts()
            top_values = value_counts.head(5)
            top_rows = [
                {
                    "value": str(k),
                    "count": str(int(v)),
                    "pct": f"{(int(v) / max(1, len(non_null))) * 100:.1f}",
                }
                for k, v in top_values.items()
            ]

            hist_bars: List[Dict[str, Any]] = []
            hist_x_min = ""
            hist_x_max = ""
            if is_numeric:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if not numeric.empty:
                    min_val = float(numeric.min())
                    max_val = float(numeric.max())
                    numeric_min = f"{min_val:.4g}"
                    numeric_max = f"{max_val:.4g}"
                    hist_x_min = numeric_min
                    hist_x_max = numeric_max
                    numeric_std = f"{float(numeric.std(ddof=0)):.4g}"
                    numeric_mean = f"{float(numeric.mean()):.4g}"
                    numeric_median = f"{float(numeric.median()):.4g}"
                    numeric_range = f"{(max_val - min_val):.4g}"
                    try:
                        counts, _bins = pd.cut(
                            numeric,
                            bins=16,
                            include_lowest=True,
                            duplicates="drop",
                            retbins=True,
                        )
                        bin_counts = counts.value_counts().sort_index()
                        for interval, count in bin_counts.items():
                            if pd.isna(interval):
                                continue
                            try:
                                ileft = float(interval.left)
                                iright = float(interval.right)
                            except Exception:
                                continue
                            hist_bars.append(
                                {
                                    "left": f"{ileft:.4g}",
                                    "right": f"{iright:.4g}",
                                    "bin": f"{ileft:.4g} - {iright:.4g}",
                                    "count": int(count),
                                }
                            )
                        if hist_bars:
                            max_count = max(1, max(int(b["count"]) for b in hist_bars))
                            for b in hist_bars:
                                h_pct = int((int(b["count"]) / max_count) * 100)
                                b["height_pct"] = str(max(8, h_pct))
                    except Exception:
                        hist_bars = []

            regex_counts: Dict[str, int] = {}
            sampled = non_null.astype(str).head(3000)
            for raw in sampled:
                sig = _regex_signature(raw)
                regex_counts[sig] = regex_counts.get(sig, 0) + 1
            regex_sorted = sorted(regex_counts.items(), key=lambda x: x[1], reverse=True)
            regex_rows: List[Dict[str, str]] = []
            sample_len = max(1, len(sampled))
            for p, c in regex_sorted:
                pct_val = (c / sample_len) * 100.0
                if pct_val < 20.0:
                    continue
                regex_rows.append(
                    {
                        "pattern": p,
                        "count": str(c),
                        "pct": f"{pct_val:.1f}",
                    }
                )
                if len(regex_rows) >= 3:
                    break
            if not regex_rows and sample_len > 0:
                # Fallback: summarize variable-length string-like columns as a broad pattern.
                alpha_mask = sampled.str.match(r"^[A-Za-z]+$", na=False)
                alpha_count = int(alpha_mask.sum())
                alpha_pct = (alpha_count / sample_len) * 100.0
                if alpha_pct >= 20.0:
                    alpha_lengths = sampled[alpha_mask].str.len()
                    if not alpha_lengths.empty:
                        min_len = int(alpha_lengths.min())
                        max_len = int(alpha_lengths.max())
                        regex_rows.append(
                            {
                                "pattern": rf"^[A-Za-z]{{{min_len},{max_len}}}$",
                                "count": str(alpha_count),
                                "pct": f"{alpha_pct:.1f}",
                            }
                        )
                else:
                    non_empty = sampled[sampled.str.len() > 0]
                    if not non_empty.empty:
                        min_len = int(non_empty.str.len().min())
                        max_len = int(non_empty.str.len().max())
                        covered = int(non_empty.shape[0])
                        pct_val = (covered / sample_len) * 100.0
                        if pct_val >= 20.0:
                            regex_rows.append(
                                {
                                    "pattern": rf"^.{{{min_len},{max_len}}}$",
                                    "count": str(covered),
                                    "pct": f"{pct_val:.1f}",
                                }
                            )

            # Labeled cells (from GUI few-shot labels) – include sample values.
            labeled_clean_rows: List[Dict[str, str]] = []
            labeled_dirty_rows: List[Dict[str, str]] = []
            label_map = dict(self.cell_labels.get(col) or {})
            if label_map:
                def _rk_sort(k: str) -> tuple:
                    try:
                        return (0, int(k))
                    except Exception:
                        return (1, k)

                for rk in sorted(label_map.keys(), key=_rk_sort):
                    entry = label_map.get(rk) or {}
                    kind = str(entry.get("kind", "") or "").lower()
                    expected = str(entry.get("expected", "") or "")
                    value = self._get_cell_at_row_id(str(rk), col)
                    rec = {
                        "row_key": str(rk),
                        "value": value,
                        "expected": expected,
                    }
                    if kind == "clean":
                        labeled_clean_rows.append(rec)
                    elif kind == "dirty":
                        labeled_dirty_rows.append(rec)

            rows.append(
                {
                    "column": col,
                    "semantic_type": semantic_type,
                    "missing_count": str(missing_count),
                    "missing_pct": f"{(missing_count / n_rows) * 100:.1f}",
                    "unique_count": str(unique_count),
                    "unique_pct": f"{uniqueness_ratio * 100:.1f}",
                    "is_numeric": "1" if is_numeric else "0",
                    "numeric_min": numeric_min,
                    "numeric_max": numeric_max,
                    "numeric_std": numeric_std,
                    "numeric_mean": numeric_mean,
                    "numeric_median": numeric_median,
                    "numeric_range": numeric_range,
                    "top_values": top_rows,
                    "histogram": hist_bars,
                    "hist_x_min": hist_x_min,
                    "hist_x_max": hist_x_max,
                    "top_regex": regex_rows,
                    "has_top_regex": "1" if len(regex_rows) > 0 else "0",
                    "color": self.column_header_colors.get(col, "#6b7280"),
                    "labeled_clean_count": str(
                        sum(
                            1
                            for v in (self.cell_labels.get(col) or {}).values()
                            if str((v or {}).get("kind", "")).lower() == "clean"
                        )
                    ),
                    "labeled_dirty_count": str(
                        sum(
                            1
                            for v in (self.cell_labels.get(col) or {}).values()
                            if str((v or {}).get("kind", "")).lower() == "dirty"
                        )
                    ),
                    "labeled_clean_rows": labeled_clean_rows,
                    "labeled_dirty_rows": labeled_dirty_rows,
                    "user_marked_clean": "1" if bool((self.user_marked_clean_columns or {}).get(col, False)) else "0",
                }
            )
        self.profiling_dashboard_rows = rows
        if not self.selected_profile_column and rows:
            self.selected_profile_column = str(rows[0]["column"])

    def _user_validation_callback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_id = str(payload.get("request_id", "") or f"{payload.get('column', '')}::{payload.get('attempt', 0)}::{int(time.time() * 1000)}")
        payload = dict(payload)
        payload["request_id"] = request_id
        with _USER_VALIDATION_CONDITION:
            _USER_VALIDATION_RESULTS.pop(request_id, None)
            _PENDING_USER_VALIDATIONS[request_id] = payload
        # Busy-wait with short sleeps to keep implementation thread-safe.
        while True:
            with _USER_VALIDATION_CONDITION:
                if request_id in _USER_VALIDATION_RESULTS:
                    result = dict(_USER_VALIDATION_RESULTS.pop(request_id))
                    return result
            global _CANCEL_EVENT
            if _CANCEL_EVENT is not None and _CANCEL_EVENT.is_set():
                with _USER_VALIDATION_CONDITION:
                    _PENDING_USER_VALIDATIONS.pop(request_id, None)
                return {"needs_correction": False, "feedback_target": "RECOMMENDER", "feedback": ""}
            time.sleep(0.25)

    def _hitl_callback_blocking(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_id = str(payload.get("request_id", "") or f"hitl::{int(time.time() * 1000)}")
        payload = dict(payload)
        payload["request_id"] = request_id
        kind = str(payload.get("kind", "") or "")
        col = str(payload.get("column", "") or "")
        with _HITL_CONDITION:
            _HITL_RESULTS.pop(request_id, None)
            _HITL_PENDING[request_id] = payload
        while True:
            with _HITL_CONDITION:
                if request_id in _HITL_RESULTS:
                    result = dict(_HITL_RESULTS.pop(request_id))
                    return result
            global _CANCEL_EVENT
            if _CANCEL_EVENT is not None and _CANCEL_EVENT.is_set():
                with _HITL_CONDITION:
                    _HITL_PENDING.pop(request_id, None)
                if kind == "code_review":
                    return {"code": str(payload.get("code", "") or "")}
                if kind == "validation_review":
                    return {"decision": "feedback_accept"}
                if kind == "already_clean_review":
                    return {"decision": "confirm"}
                return {}
            time.sleep(0.25)

    @staticmethod
    def _parse_optional_float(s: str) -> Optional[float]:
        t = (s or "").strip()
        if not t:
            return None
        try:
            return float(t)
        except ValueError:
            return None

    @staticmethod
    def _normalize_sample_row(r: Dict[str, Any]) -> Dict[str, str]:
        o = str(r.get("original", "") or "")
        c = str(r.get("cleaned", "") or "")
        ch = str(r.get("changed", "") or "")
        if ch not in ("0", "1"):
            ch = "1" if o != c else "0"
        return {"original": o, "cleaned": c, "changed": ch}

    def _reset_hitl_session_state(self) -> None:
        self.hitl_last_synced_rid_by_column = {}
        self.hitl_code_by_column = {}
        self.hitl_validation_feedback_target_by_column = {}
        self.hitl_validation_feedback_message_by_column = {}

    def _hitl_rid_for_column_kind(self, column: str, kind: str) -> str:
        with _HITL_CONDITION:
            for rid, p in _HITL_PENDING.items():
                if str(p.get("column", "") or "") == column and str(p.get("kind", "") or "") == kind:
                    return str(rid)
        return ""

    def _resolve_hitl_request_id(self, request_id: str, column: str, kind: str) -> str:
        rid = str(request_id or "").strip()
        if rid:
            with _HITL_CONDITION:
                if rid in _HITL_PENDING:
                    return rid
            # In parallel HITL mode, silently remapping a missing request_id can resolve
            # a different pending request while leaving the intended request blocked.
            return ""
        fallback = self._hitl_rid_for_column_kind(column, kind)
        return fallback

    def _finish_hitl_for_request(self, rid: str, payload: Dict[str, Any], status_msg: str) -> None:
        if not rid:
            return
        with _HITL_CONDITION:
            pending = _HITL_PENDING.get(rid, {})
            kind = str(pending.get("kind", "") or "")
            col = str(pending.get("column", "") or "")
            _HITL_RESULTS[rid] = payload
            _HITL_PENDING.pop(rid, None)
            _HITL_CONDITION.notify_all()
        self._rebuild_pipeline_flow_rows()
        self.selected_trace_status = status_msg

    def set_human_in_the_loop(self, val: bool):
        self.human_in_the_loop = bool(val)

    def set_hitl_apply_to_all_columns(self, val: bool):
        self.hitl_apply_to_all_columns = bool(val)
        if self.column_names:
            self._rebuild_hitl_column_checkbox_rows()

    def set_llm_temperature_input(self, val: str):
        self.llm_temperature_input = val

    def set_llm_top_p_input(self, val: str):
        self.llm_top_p_input = val

    def set_selected_llm_key_all_agents(self, val: str):
        self.selected_llm_key_recommender = val
        self.selected_llm_key_coding = val
        self.selected_llm_key_validation = val
        self.enable_user_validation = val == "USER"

    @rx.event
    def set_hitl_code_column(self, column: str, value: str):
        m = dict(self.hitl_code_by_column or {})
        m[column] = value
        self.hitl_code_by_column = m

    @rx.event
    def set_hitl_validation_feedback_target_column(self, column: str, val: str):
        if val not in ("CODER", "RECOMMENDER"):
            return
        m = dict(self.hitl_validation_feedback_target_by_column or {})
        m[column] = val
        self.hitl_validation_feedback_target_by_column = m

    @rx.event
    def set_hitl_validation_feedback_message_column(self, column: str, val: str):
        m = dict(self.hitl_validation_feedback_message_by_column or {})
        m[column] = val
        self.hitl_validation_feedback_message_by_column = m

    def set_recommender_hint_editor_text(self, val: str):
        self.recommender_hint_editor_text = val

    def _rebuild_hitl_column_checkbox_rows(self) -> None:
        sel = set(self.hitl_selected_columns or [])
        order = list(self.column_names or [])
        self.hitl_column_checkbox_rows = [
            {"column": c, "active": "1" if c in sel else "0"} for c in order
        ]

    @rx.event
    def toggle_hitl_column_pick(self, column: str):
        s = set(self.hitl_selected_columns or [])
        if column in s:
            s.discard(column)
        else:
            s.add(column)
        order = list(self.column_names or [])
        self.hitl_selected_columns = [c for c in order if c in s]
        self._rebuild_hitl_column_checkbox_rows()

    @rx.event
    def open_recommender_hint_editor(self, col: str):
        self.recommender_hint_editing_column = col
        self.recommender_hint_editor_text = str((self.recommender_column_hints or {}).get(col, "") or "")
        self.recommender_hint_dialog_open = True

    @rx.event
    def save_recommender_hint_and_close(self):
        c = (self.recommender_hint_editing_column or "").strip()
        self.recommender_hint_dialog_open = False
        if not c:
            return
        text = (self.recommender_hint_editor_text or "").strip()
        h = dict(self.recommender_column_hints or {})
        if text:
            h[c] = text
        else:
            h.pop(c, None)
        self.recommender_column_hints = h

    @rx.event
    def cancel_recommender_hint_editor(self):
        self.recommender_hint_dialog_open = False

    def set_label_cells_mode(self, val: bool):
        self.label_cells_mode = bool(val)
        if not self.label_cells_mode:
            self.labeling_dialog_open = False

    def _rebuild_display_column_header_colors(self) -> None:
        base = dict(self.column_header_colors or {})
        out: Dict[str, str] = {}
        for col in self.column_names:
            color = base.get(col, "#6b7280")
            # Header colors should not react to per-cell labels; only explicit
            # per-column "user marked clean" can override the semantic color.
            if bool((self.user_marked_clean_columns or {}).get(col, False)):
                color = "#15803d"
            out[col] = color
        self.display_column_header_colors = out

    def _label_flags_full_row(self, row_id: str) -> List[bool]:
        return [row_id in (self.cell_labels.get(c) or {}) for c in self.column_names]

    def _label_kinds_full_row(self, row_id: str) -> List[str]:
        kinds: List[str] = []
        for col in self.column_names:
            entry = (self.cell_labels.get(col) or {}).get(row_id) or {}
            kind = str(entry.get("kind") or "").strip().lower()
            kinds.append(kind if kind in ("clean", "dirty") else "")
        return kinds

    def _get_cell_at_row_id(self, row_id: str, col: str) -> str:
        if self._full_df is None or col not in self._full_df.columns:
            return ""
        try:
            pos = int(str(row_id).strip())
        except Exception:
            return ""
        if pos < 0 or pos >= len(self._full_df.index):
            return ""
        try:
            v = self._full_df.iloc[pos][col]
            return "" if pd.isna(v) else str(v)
        except Exception:
            return ""
        return ""

    def _sync_labeled_examples_to_config(self) -> None:
        new_map: Dict[str, str] = {}
        for col, rows in (self.cell_labels or {}).items():
            if col not in self.column_names or not rows:
                continue
            lines: List[str] = []

            def _rk(k: str) -> tuple:
                try:
                    return (0, int(k))
                except ValueError:
                    return (1, k)

            selected_row_keys = list(sorted(rows.keys(), key=_rk))
            limit = max(1, int(self.max_labeled_cells_per_column or 1))
            if len(selected_row_keys) > limit:
                selected_row_keys = sorted(random.sample(selected_row_keys, limit), key=_rk)

            for row_key in selected_row_keys:
                entry = rows[row_key]
                kind = (entry.get("kind") or "clean").lower()
                if kind == "dirty":
                    exp = entry.get("expected", "")
                    lines.append(
                        f"- Row index {row_key}: DIRTY — after cleaning, the cell value must become: {repr(exp)}"
                    )
                else:
                    lines.append(
                        f"- Row index {row_key}: CLEAN — the value is already acceptable; do not unnecessarily rewrite this cell."
                    )
            new_map[col] = "\n".join(lines)
        self.recommender_column_labeled_examples = new_map

    @rx.event
    def open_cell_label_dialog(self, row_id: Any, col: str, current_value: Any = ""):
        if not self.label_cells_mode:
            return
        col = str(col).strip()
        if not col or col not in self.column_names:
            return
        rk = str(row_id).strip()
        self.labeling_target_col = col
        self.labeling_target_row_key = rk
        # Prefer the value from the clicked table cell; fallback to DataFrame lookup.
        current_value_str = ""
        try:
            if current_value is None or pd.isna(current_value):
                current_value_str = ""
            else:
                current_value_str = str(current_value)
        except Exception:
            current_value_str = str(current_value or "")
        if current_value_str.strip().lower() in ("nan", "none"):
            current_value_str = ""
        resolved_preview = (
            current_value_str if current_value_str != "" else self._get_cell_at_row_id(rk, col)
        )
        self.labeling_current_value_preview = resolved_preview
        existing = (self.cell_labels.get(col) or {}).get(rk)
        if existing:
            self.labeling_kind = (existing.get("kind") or "clean").lower()
            if self.labeling_kind not in ("clean", "dirty"):
                self.labeling_kind = "clean"
            self.labeling_expected_value = str(existing.get("expected") or "")
        else:
            self.labeling_kind = "clean"
            self.labeling_expected_value = ""
        self.labeling_dialog_open = True

    @rx.event
    def open_cell_label_dialog_by_position(self, evt_or_key: Any, pos_key: Any = ""):
        """Open label dialog by packed 'row:col' coordinates; tolerant of injected click event arg."""
        event_obj = evt_or_key if isinstance(evt_or_key, dict) else None
        packed = str(pos_key if event_obj is not None else evt_or_key).strip()
        if not self.label_cells_mode:
            return
        if ":" not in packed:
            return
        left, right = packed.split(":", 1)
        try:
            rpos = int(left)
            cidx = int(right)
        except Exception:
            return
        if rpos < 0 or cidx < 0:
            return
        if cidx >= len(self.column_names):
            return
        if rpos >= len(self.df_preview):
            return
        row = self.df_preview[rpos] or {}
        col = str(self.column_names[cidx])
        rk = str(row.get("__row_id", str(self.page_offset + rpos)))
        current_value = row.get(col)
        self.open_cell_label_dialog(rk, col, current_value)

    @rx.event
    def save_cell_label(self):
        col = (self.labeling_target_col or "").strip()
        rk = (self.labeling_target_row_key or "").strip()
        self.labeling_dialog_open = False
        if not col or rk == "" or col not in self.column_names:
            return
        self.cell_labels.setdefault(col, {})
        if self.labeling_kind == "dirty":
            self.cell_labels[col][rk] = {
                "kind": "dirty",
                "expected": (self.labeling_expected_value or "").strip(),
            }
        else:
            self.cell_labels[col][rk] = {"kind": "clean", "expected": ""}
        self._sync_labeled_examples_to_config()
        self._rebuild_display_column_header_colors()
        self._build_profiling_dashboard_rows()
        self._refresh_current_page()

    @rx.event
    def clear_current_cell_label(self):
        col = (self.labeling_target_col or "").strip()
        rk = (self.labeling_target_row_key or "").strip()
        self.labeling_dialog_open = False
        if col in self.cell_labels and rk in self.cell_labels[col]:
            del self.cell_labels[col][rk]
            if not self.cell_labels[col]:
                del self.cell_labels[col]
        self._sync_labeled_examples_to_config()
        self._rebuild_display_column_header_colors()
        self._build_profiling_dashboard_rows()
        self._refresh_current_page()

    @rx.event
    def cancel_cell_label_dialog(self):
        self.labeling_dialog_open = False

    @rx.event
    def toggle_user_marked_column_clean(self, col: str):
        col = str(col).strip()
        if not col or col not in self.column_names:
            return
        m = dict(self.user_marked_clean_columns or {})
        m[col] = not bool(m.get(col, False))
        self.user_marked_clean_columns = m
        self._rebuild_display_column_header_colors()
        self._build_profiling_dashboard_rows()
        self._update_pipeline_trace_view()

    def set_labeling_kind(self, val: str):
        self.labeling_kind = "dirty" if str(val).lower().strip() == "dirty" else "clean"

    def set_labeling_expected_value(self, val: str):
        self.labeling_expected_value = val

    @rx.event
    def submit_hitl_code_review(self, request_id: str, column: str):
        request_id = self._resolve_hitl_request_id(request_id, column, "code_review")
        code = (self.hitl_code_by_column or {}).get(column, "")
        self._finish_hitl_for_request(request_id, {"code": code}, "Code review submitted. Pipeline resumed.")

    def _submit_hitl_validation_decision_request(self, request_id: str, column: str, decision: str) -> None:
        request_id = self._resolve_hitl_request_id(request_id, column, "validation_review")
        if not request_id:
            return
        payload: Dict[str, Any] = {"decision": decision}
        if decision in ("validator_ok_disagree", "feedback_reject_revise"):
            ft = (self.hitl_validation_feedback_target_by_column or {}).get(column, "RECOMMENDER")
            if ft not in ("CODER", "RECOMMENDER"):
                ft = "RECOMMENDER"
            payload["feedback_target"] = ft
            payload["correction_instructions"] = (self.hitl_validation_feedback_message_by_column or {}).get(column, "")
        self._finish_hitl_for_request(request_id, payload, "Validation decision submitted. Pipeline resumed.")

    @rx.event
    def submit_hitl_validation_validator_ok_agree(self, request_id: str, column: str):
        self._submit_hitl_validation_decision_request(request_id, column, "validator_ok_agree")

    @rx.event
    def submit_hitl_validation_validator_ok_disagree(self, request_id: str, column: str):
        self._submit_hitl_validation_decision_request(request_id, column, "validator_ok_disagree")

    @rx.event
    def submit_hitl_validation_feedback_accept(self, request_id: str, column: str):
        self._submit_hitl_validation_decision_request(request_id, column, "feedback_accept")

    @rx.event
    def submit_hitl_validation_feedback_reject_cleaning_valid(self, request_id: str, column: str):
        self._submit_hitl_validation_decision_request(request_id, column, "feedback_reject_cleaning_valid")

    @rx.event
    def submit_hitl_validation_feedback_reject_revise(self, request_id: str, column: str):
        self._submit_hitl_validation_decision_request(request_id, column, "feedback_reject_revise")

    @rx.event
    def submit_hitl_already_clean_confirm(self, request_id: str, column: str):
        request_id = self._resolve_hitl_request_id(request_id, column, "already_clean_review")
        self._finish_hitl_for_request(request_id, {"decision": "confirm"}, "Already-clean check submitted. Pipeline resumed.")

    @rx.event
    def submit_hitl_already_clean_reject(self, request_id: str, column: str):
        request_id = self._resolve_hitl_request_id(request_id, column, "already_clean_review")
        msg = (self.hitl_validation_feedback_message_by_column or {}).get(column, "")
        self._finish_hitl_for_request(
            request_id,
            {"decision": "reject", "rejection_reason": msg},
            "Already-clean check submitted. Pipeline resumed.",
        )

    # ============================================================
    # PROFILING
    # ============================================================
    @rx.event(background=True)
    async def run_profiling_process(self):
        if self._full_df is None or self._full_df.empty:
            async with self:
                self.profiling_status = "No data to profile."
            return

        async with self:
            self.is_profiling = True
            self.profiling_status = "Profiling dataset..."
            self.column_semantic_types = {}
            self.fd_results = []
            self.selected_fd_keys = []
            self.column_fd_markers = {}

        config = CleaningConfig(
            verbose=self.verbose,
            enable_validation=self.enable_validation,
            enable_user_validation=self.selected_llm_key_validation == "USER",
            enable_validation_multi=self.enable_validation_multi,
            enable_multi_col_cleaning=self.enable_multi_col_cleaning,
            sample_sizes=self.sample_sizes,
            sample_size_validator=self.sample_size_validator,
            sample_size_validator_random=self.sample_size_validator_random,
            sample_size_validator_changed=self.sample_size_validator_changed,
            validator_failure_strategy=self.validator_failure_strategy,
            max_cleaning_attempts=self.max_cleaning_attempts,
            max_multi_col_attempts=self.max_multi_col_attempts,
            max_parse_attempts=self.max_parse_attempts,
            max_coding_attempts=self.max_coding_attempts,
            semaphore_limit=self.semaphore_limit,
            include_metadata=self.include_metadata,
            skip_columns=[
                c
                for c, is_clean in (self.user_marked_clean_columns or {}).items()
                if bool(is_clean)
            ],
        )
        profiler = DataProfiler(
            single_col_cleaners=[OutlierDetection()],
            multi_col_cleaners=[FunctionalDependencies()],
            config=config,
        )

        loop = asyncio.get_event_loop()
        profiles_and_tasks = await loop.run_in_executor(None, profiler.analyse, self._full_df)
        profiles, multi_col_tasks = profiles_and_tasks

        semantic_types: dict[str, str] = {}
        fd_rows: list[dict[str, Any]] = []
        for col, profile in (profiles or {}).items():
            if isinstance(profile, ColumnProfile):
                semantic_types[col] = profile.semantic_type
            else:
                semantic_types[col] = str(getattr(profile, "semantic_type", "UNKNOWN"))

        for task in (multi_col_tasks or []):
            if not isinstance(task, MultiColumnTask):
                continue
            if task.task_type != "FD":
                continue
            fd = task.data
            if isinstance(fd, FDResult):
                fd_rows.append(
                    {
                        "lhs": fd.lhs,
                        "rhs": fd.rhs,
                        "score": fd.score,
                        "violations_count": fd.violations_count,
                        "imputables_count": fd.imputables_count,
                    }
                )
            else:
                fd_rows.append(
                    {
                        "lhs": task.target_columns[0] if task.target_columns else "",
                        "rhs": task.target_columns[1] if len(task.target_columns) > 1 else "",
                        "score": None,
                        "violations_count": None,
                        "imputables_count": None,
                    }
                )

        async with self:
            self.column_semantic_types = semantic_types
            self.column_header_colors = {
                col: self._semantic_type_to_color(semantic_types.get(col, "UNKNOWN"))
                for col in self.column_names
            }
            self._rebuild_display_column_header_colors()
            self.column_profile_rows = [
                {
                    "name": col,
                    "semantic_type": self.column_semantic_types.get(col, "UNKNOWN"),
                    "color": self.column_header_colors.get(col, "#9CA3AF"),
                }
                for col in self.column_names
            ]
            self.fd_results = fd_rows
            self._rebuild_fd_graph_rows()
            self._rebuild_fd_button_rows()
            self._rebuild_fd_header_badges()
            self._rebuild_column_fd_markers()
            self._rebuild_selected_fd_arrows()
            self._build_profiling_dashboard_rows()
            self.profiling_status = f"Profiled {len(semantic_types)} columns, found {len(fd_rows)} FDs with issues."
            self.is_profiling = False

    @rx.event(background=True)
    async def run_cleaning_process(self):
        if not self._file_path:
            async with self: self.status_msg = "Error: No file selected."
            return

        # Prevent double-starting the pipeline.
        async with self:
            if self.is_cleaning:
                self.status_msg = "Pipeline already running."
                return
        
        async with self:
            self.is_cleaning = True
            self.last_run_status = "running"
            self.status_msg = "Initializing Pipeline..."
            self.progress_percent = 0
            self.runtime_seconds = 0.0
            self.runtime_seconds_display = "0.0"
            self.token_usage = {}
            self.token_usage_pretty = "{}"
            self.token_usage_block = ""
            self.token_usage_donut_rows = []
            self.cleaning_summary = {}
            self.modified_cell_keys = []
            self.logs = []
            self._log_file_offset = 0
            self._trace_file_offset = 0
            self._log_file_path = os.path.join(rx.get_upload_dir(), "madclean_run_logs.txt")
            self._trace_file_path = os.path.join(rx.get_upload_dir(), "madclean_trace_events.jsonl")
            self.column_pipeline_traces = {}
            self.pipeline_trace_columns = []
            self.active_trace_step_by_column = {}
            self.expanded_trace_step_by_column = {}
            self.pipeline_flow_rows = []
            self.selected_trace_status = "Cleaning started. Waiting for live workflow events..."
            self.pending_user_validation = False
            self.pending_validation_request_ids = []
            self.pending_validation_request_id_by_column = {}
            self.pending_validation_column = ""
            self.pending_validation_attempt = 0
            self.pending_validation_sample_rows = []
            self.pending_validation_modified_count = 0
            self.user_validation_feedback_message = ""
            self._update_pending_validation_widths()
            self._reset_hitl_session_state()
            with _HITL_CONDITION:
                _HITL_PENDING.clear()
                _HITL_RESULTS.clear()
            # Reset log file at the start of each run.
            try:
                with open(self._log_file_path, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception:
                self._log_file_path = ""
            try:
                with open(self._trace_file_path, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception:
                self._trace_file_path = ""
            global _CANCEL_EVENT
            _CANCEL_EVENT = threading.Event()

        config = CleaningConfig(
            verbose=self.verbose,
            enable_validation=self.enable_validation,
            enable_user_validation=self.selected_llm_key_validation == "USER",
            enable_validation_multi=self.enable_validation_multi,
            enable_multi_col_cleaning=self.enable_multi_col_cleaning,
            sample_sizes=self.sample_sizes,
            sample_size_validator=self.sample_size_validator,
            sample_size_validator_random=self.sample_size_validator_random,
            sample_size_validator_changed=self.sample_size_validator_changed,
            validator_failure_strategy=self.validator_failure_strategy,
            max_cleaning_attempts=self.max_cleaning_attempts,
            max_multi_col_attempts=self.max_multi_col_attempts,
            max_parse_attempts=self.max_parse_attempts,
            max_coding_attempts=self.max_coding_attempts,
            semaphore_limit=self.semaphore_limit,
            include_metadata=self.include_metadata,
            human_in_the_loop=self.human_in_the_loop,
            hitl_apply_to_all_columns=self.hitl_apply_to_all_columns,
            hitl_column_list=list(self.hitl_selected_columns or []),
            recommender_column_hints={
                k: (v or "").strip()
                for k, v in (self.recommender_column_hints or {}).items()
                if (v or "").strip()
            },
            recommender_column_labeled_examples={
                k: (v or "").strip()
                for k, v in (self.recommender_column_labeled_examples or {}).items()
                if (v or "").strip()
            },
            llm_temperature=self._parse_optional_float(self.llm_temperature_input),
            llm_top_p=self._parse_optional_float(self.llm_top_p_input),
            skip_columns=[
                c
                for c, is_clean in (self.user_marked_clean_columns or {}).items()
                if bool(is_clean)
            ],
        )

        llm_configs = {
            "recommender": LLM_CLIENT_MAP[self.selected_llm_key_recommender],
            "coding": LLM_CLIENT_MAP[self.selected_llm_key_coding],
            "validation": (
                LLM_CLIENT_MAP[self.selected_llm_key_validation]
                if self.selected_llm_key_validation in LLM_CLIENT_MAP
                else LLM_CLIENT_MAP[self.selected_llm_key_coding]
            ),
        }
        def _cancel_check():
            global _CANCEL_EVENT
            return _CANCEL_EVENT is not None and _CANCEL_EVENT.is_set()

        pipeline = Pipeline(
            llm_config=llm_configs["coding"],
            agent_llm_configs=llm_configs,
            config=config,
            log_callback=self._enqueue_log,
            trace_callback=self._enqueue_trace_event,
            user_validation_callback=self._user_validation_callback,
            hitl_callback=self._hitl_callback_blocking,
            cancel_check=_cancel_check,
        )
        coordinator = pipeline.cleaning_coordinator
        
        loop = asyncio.get_event_loop()
        start = time.perf_counter()
        task = loop.run_in_executor(None, pipeline.run, self._file_path)

        while not task.done():
            await asyncio.sleep(0.5)
            await self._drain_logs()
            await self._drain_trace_events()
            async with self:
                self.progress_percent = int(coordinator.progress.get("current_percentage", 0))
                self.status_msg = f"Cleaning... {self.progress_percent}%"
        
        result = await task
        await self._drain_logs()
        await self._drain_trace_events()
        end = time.perf_counter()
        async with self:
            if result is not None:
                cleaned_df, report = result
                # The GUI reads the report as the flat dictionary the thesis returned.
                report = report.to_dict() if report is not None else None
                cancelled = bool((report or {}).get("cancelled")) if isinstance(report, dict) else False
                if cancelled or cleaned_df is None:
                    # Keep the currently loaded dataset; do not overwrite with None.
                    self.last_run_status = "cancelled" if cancelled else "error"
                    self.has_cleaned = False
                    self.status_msg = "Cancelled" if cancelled else "Pipeline Error."
                    self.logs = ["Pipeline stopped. No cleaning result was generated."]
                    self.runtime_seconds = 0.0
                    self.runtime_seconds_display = "0.0"
                    self.token_usage = {}
                    self.token_usage_pretty = "{}"
                    self.token_usage_block = "Pipeline stopped. Token usage is not available for this run."
                    self.token_usage_donut_rows = []
                    self.cleaning_summary = {}
                    self.report_code_keys = []
                    self.selected_report_code_key = ""
                    self.generated_code_for_selected = ""
                    self.generated_code_for_selected_pretty = ""
                    self.generated_code_lines_for_selected = []
                    self.selected_report_meta = {}
                    self.selected_report_meta_pretty = "{}"
                    self.selected_report_meta_lines = []
                    self.selected_report_status = "Pipeline stopped. No report generated."
                    self._run_evaluation()
                else:
                    self._full_df = cleaned_df
                    self.has_cleaned = True
                    self.last_run_status = "finished"
                    self.total_rows = int(len(cleaned_df))
                    self.modified_cell_keys = []
                    self.token_usage = (report or {}).get("token_usage", {})
                    try:
                        self.token_usage_pretty = json.dumps(self.token_usage, indent=2, default=str)
                    except Exception:
                        self.token_usage_pretty = str(self.token_usage)
                    # Donut charts now carry the main token-usage visualization; keep block empty.
                    self.token_usage_block = ""
                    self._build_token_usage_donut_rows(report or {})
                    self.runtime_seconds = float((report or {}).get("runtime_seconds", end - start))
                    self.runtime_seconds_display = f"{self.runtime_seconds:.1f}"
                    self.cleaning_summary = report or {}
                    self._update_report_code_view(self.cleaning_summary)
                    self._update_pipeline_trace_view()
                    self._build_profiling_dashboard_rows()
                    self.page_offset = 0
                    self._refresh_current_page()
                    self.status_msg = "Finished"
                    self._run_evaluation()
            else:
                self.status_msg = "Pipeline Error."
                self.last_run_status = "error"
            self.is_cleaning = False
            # keep log file path for post-mortem/debugging

    async def handle_upload(self, files: List[rx.UploadFile]):
        if not files: return
        file = files[0]
        upload_data = await file.read()
        upload_dir = rx.get_upload_dir()
        if not os.path.exists(upload_dir): os.makedirs(upload_dir)
        outfile = os.path.join(upload_dir, file.filename)
        with open(outfile, "wb") as f: f.write(upload_data)
        try:
            df = pd.read_csv(outfile) if file.filename.endswith(".csv") else pd.read_excel(outfile)
            async with self:
                self._orig_df = df.copy()
                self._full_df = df
                self._file_path = outfile
                self.file_name = file.filename
                self.column_names = list(df.columns)
                self.has_cleaned = False
                self.selected_main_tab = "table"
                self.total_rows = int(len(df))
                self.page_offset = 0
                self._refresh_current_page()
                self.modified_cell_keys = []
                self.token_usage = {}
                self.token_usage_donut_rows = []
                self.logs = []
                self.cleaning_summary = {}
                self.report_code_keys = []
                self.selected_report_code_key = ""
                self.generated_code_for_selected = ""
                self.selected_report_meta = {}
                self.fd_results = []
                self.fd_graph_rows = []
                self.show_fd_graph = False
                self.selected_fd_keys = []
                self.fd_button_rows = []
                self.fd_header_badges = {}
                self.column_fd_markers = {}
                self.selected_fd_arrows = []
                self.column_pipeline_traces = {}
                self.pipeline_trace_columns = []
                self.active_trace_step_by_column = {}
                self.expanded_trace_step_by_column = {}
                self.pipeline_flow_rows = []
                self.selected_trace_status = "Run cleaning to view live column workflows."
                self.pending_user_validation = False
                self.pending_validation_request_ids = []
                self.pending_validation_request_id_by_column = {}
                self.pending_validation_column = ""
                self.pending_validation_attempt = 0
                self.pending_validation_sample_rows = []
                self.pending_validation_modified_count = 0
                self.user_validation_feedback_message = ""
                self._update_pending_validation_widths()
                # Initialize profiling UI defaults (runs after upload).
                self.column_semantic_types = {c: "UNKNOWN" for c in self.column_names}
                self.column_header_colors = {c: self._semantic_type_to_color("UNKNOWN") for c in self.column_names}
                self._rebuild_display_column_header_colors()
                self.column_profile_rows = [
                    {"name": c, "semantic_type": "UNKNOWN", "color": self.column_header_colors[c]}
                    for c in self.column_names
                ]
                self.profiling_dashboard_rows = []
                self.selected_profile_column = self.column_names[0] if self.column_names else ""
                self.regex_pattern_input = ""
                self.regex_results_rows = []
                self.regex_status = "Enter a regex and run detection."
                self.profiling_status = "Waiting for profiling..."
                self.is_profiling = False
                self.status_msg = f"Loaded {file.filename}"
                cols = list(self.column_names or [])
                self.recommender_column_hints = {
                    k: v for k, v in (self.recommender_column_hints or {}).items() if k in cols
                }
                self.cell_labels = {}
                self.user_marked_clean_columns = {}
                self.label_cells_mode = False
                self.labeling_dialog_open = False
                self._sync_labeled_examples_to_config()
                self._rebuild_display_column_header_colors()
                self.hitl_selected_columns = [c for c in (self.hitl_selected_columns or []) if c in cols]
                self._rebuild_hitl_column_checkbox_rows()
                self.recommender_hint_dialog_open = False
                self._gt_df = None
                self.gt_file_name = "No ground-truth file"
                self._run_evaluation()
            # Kick off profiling in background (don’t block upload).
            # Reflex requires using `yield State.run_profiling_process` for background tasks.
            yield State.run_profiling_process
        except Exception as e:
            async with self:
                self.status_msg = f"Upload Error: {str(e)}"

    async def handle_gt_upload(self, files: List[rx.UploadFile]):
        if not files:
            return
        file = files[0]
        upload_data = await file.read()
        upload_dir = rx.get_upload_dir()
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        safe_name = f"gt_{file.filename}"
        outfile = os.path.join(upload_dir, safe_name)
        with open(outfile, "wb") as f:
            f.write(upload_data)
        try:
            df = (
                pd.read_csv(outfile)
                if file.filename.lower().endswith(".csv")
                else pd.read_excel(outfile)
            )
            async with self:
                self._gt_df = df
                self.gt_file_name = file.filename
                self._run_evaluation()
                if self.evaluation_error:
                    self.status_msg = "Ground truth incompatible or error — see Evaluation tab."
                else:
                    self.status_msg = f"Ground truth loaded: {file.filename}"
        except Exception as e:
            async with self:
                self._gt_df = None
                self.gt_file_name = "No ground-truth file"
                self.evaluation_error = str(e)
                self.evaluation_status = "Failed to parse ground-truth file."
                self.evaluation_ready = False
                self.evaluation_compatible = False
                self.evaluation_overall_rows = []
                self.evaluation_column_rows = []
                self.status_msg = f"Ground truth upload error: {str(e)}"

    def download_cleaned_file(self):
        if self._full_df is not None:
            return rx.download(data=self._full_df.to_csv(index=False), filename=f"cleaned_{self.file_name}")

    def download_cleaning_code(self):
        code = (self.merged_generated_code or "").strip()
        if not code:
            return
        base_name = self.file_name.rsplit(".", 1)[0] if self.file_name and "." in self.file_name else "dataset"
        return rx.download(data=code, filename=f"cleaning_code_{base_name}.py")