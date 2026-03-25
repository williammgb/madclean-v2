import reflex as rx
import pandas as pd
import asyncio
import os
import time
import json
import threading
import math
from typing import Optional, List, Dict, Any, Callable
from madclean.config.settings import CleaningConfig
from madclean.llm.llm_registry import LLM_CLIENT_MAP
from madclean.pipeline import Pipeline
from madclean.components.domain.schema import MultiColumnTask, ColumnProfile, FDResult
from madclean.components.dataprofiler.dataprofiler import DataProfiler
from madclean.components.dataprofiler.outlier_detection import OutlierDetection
from madclean.components.dataprofiler.functional_dependencies import FunctionalDependencies

_CANCEL_EVENT: threading.Event | None = None

class State(rx.State):
    """Bridge between MADClean system and the UI."""
    _default_config = CleaningConfig()

    # Configuration State
    verbose: bool = _default_config.verbose
    enable_validation: bool = _default_config.enable_validation
    enable_validation_multi: bool = _default_config.enable_validation_multi
    enable_multi_col_cleaning: bool = _default_config.enable_multi_col_cleaning
    
    # Nested Dictionary for Sample Sizes
    sample_sizes: Dict[str, Dict[str, int]] = {
        "NUMERIC": {"clean_sample_size": 50, "dirty_sample_size": 500},
        "DATETIME": {"clean_sample_size": 100, "dirty_sample_size": 500},
        "DIRTY_NUMERIC": {"random_sample_size": 50, "unique_sample_size": 500},
        "STRING": {"random_sample_size": 150, "unique_sample_size": 600},
        "NLT": {"short_sample_size": 100, "long_sample_size": 20}
    }

    sample_size_validator: int = _default_config.sample_size_validator
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
    _max_logs: int = 500
    
    # LLM Settings
    llm_options: List[str] = list(LLM_CLIENT_MAP.keys())
    # Per-agent model selection (defaults to the first configured model).
    selected_llm_key_recommender: str = llm_options[0] if llm_options else ""
    selected_llm_key_coding: str = llm_options[0] if llm_options else ""
    selected_llm_key_validation: str = llm_options[0] if llm_options else ""

    # Data State
    file_name: str = "No file selected"
    _file_path: str = "" 
    df_preview: List[Dict[str, Any]] = []
    column_names: List[str] = []
    _full_df: Optional[pd.DataFrame] = None
    _orig_df: Optional[pd.DataFrame] = None
    has_cleaned: bool = False
    modified_cell_keys: List[str] = []  # debug/secondary: f"{row_id}::{col}"
    token_usage: Dict[str, Any] = {}
    token_usage_pretty: str = "{}"
    token_usage_block: str = ""
    runtime_seconds: float = 0.0
    runtime_seconds_display: str = "0.0"
    cleaning_summary: Dict[str, Any] = {}
    report_code_keys: List[str] = []
    selected_report_code_key: str = ""
    generated_code_for_selected: str = ""
    generated_code_for_selected_pretty: str = ""
    generated_code_lines_for_selected: List[str] = []
    selected_report_meta: Dict[str, Any] = {}
    selected_report_meta_pretty: str = "{}"
    selected_report_meta_lines: List[str] = []
    selected_report_status: str = ""

    column_header_colors: Dict[str, str] = {}  # col -> hex color string
    column_profile_rows: List[Dict[str, str]] = []  # [{name, semantic_type, color}]

    # Profiling state (types + FDs)
    is_profiling: bool = False
    profiling_status: str = ""
    column_semantic_types: Dict[str, str] = {}  # col -> semantic type
    fd_results: List[Dict[str, Any]] = []  # list of {lhs,rhs,score,violations_count,imputables_count}

    # Table pagination (100-row pages).
    page_offset: int = 0
    page_size: int = 100
    total_rows: int = 0

    # Bottom panel sizing (px).
    bottom_panel_height: int = 240

    # ============================================================
    # EXPLICIT SETTERS
    # ============================================================
    def set_verbose(self, val: bool): self.verbose = val
    def set_enable_validation(self, val: bool): self.enable_validation = val
    def set_enable_validation_multi(self, val: bool): self.enable_validation_multi = val
    def set_enable_multi_col_cleaning(self, val: bool): self.enable_multi_col_cleaning = val
    def set_include_metadata(self, val: bool): self.include_metadata = val
    def set_selected_llm_key_recommender(self, val: str): self.selected_llm_key_recommender = val
    def set_selected_llm_key_coding(self, val: str): self.selected_llm_key_coding = val
    def set_selected_llm_key_validation(self, val: str): self.selected_llm_key_validation = val

    def set_selected_report_code_key(self, val: str):
        self.selected_report_code_key = val
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

    @rx.event
    def stop_pipeline(self):
        """Request cooperative cancellation for the currently running pipeline."""
        global _CANCEL_EVENT
        if _CANCEL_EVENT is not None:
            _CANCEL_EVENT.set()
        # UI feedback (actual stop may take a bit while LLM calls finish).
        self.status_msg = "Stopping pipeline..."

    def _update_report_code_view(self, report: Dict[str, Any]):
        # Show all columns present in the report (including already-cleaned ones).
        code_keys: list[str] = []
        for k, v in (report or {}).items():
            if not isinstance(v, dict):
                continue
            # Only show per-column / per-task entries, not global report fields.
            if k in ("token_usage", "total_usage"):
                continue
            if k in self.column_names or "→" in k:
                code_keys.append(k)
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
    def set_max_cleaning_attempts(self, val: str): self.max_cleaning_attempts = int(val) if val.isdigit() else self.max_cleaning_attempts
    def set_max_multi_col_attempts(self, val: str): self.max_multi_col_attempts = int(val) if val.isdigit() else self.max_multi_col_attempts
    def set_max_parse_attempts(self, val: str): self.max_parse_attempts = int(val) if val.isdigit() else self.max_parse_attempts
    def set_max_coding_attempts(self, val: str): self.max_coding_attempts = int(val) if val.isdigit() else self.max_coding_attempts
    def set_semaphore_limit(self, val: str): self.semaphore_limit = int(val) if val.isdigit() else self.semaphore_limit

    @rx.event
    def prev_page(self):
        self.page_offset = max(0, self.page_offset - self.page_size)
        self._refresh_current_page()

    @rx.event
    def next_page(self):
        if self.total_rows <= 0:
            return
        max_start = max(0, self.total_rows - self.page_size)
        self.page_offset = min(max_start, self.page_offset + self.page_size)
        self._refresh_current_page()

    def set_bottom_panel_height(self, val: List[float]):
        if not val:
            return
        h = int(val[0])
        self.bottom_panel_height = max(160, min(900, h))

    @staticmethod
    def _pretty_python_code(code: str) -> str:
        if not code:
            return ""
        # Try to pretty-format using black if available.
        try:
            import black  # type: ignore

            return black.format_str(code, mode=black.FileMode())
        except Exception:
            # Fallback: normalize indentation and strip trailing spaces.
            import textwrap

            return "\n".join(line.rstrip() for line in textwrap.dedent(code).strip().splitlines())

    @rx.event
    def update_cell(self, row_id: int, col: str, value: str):
        """Edit a single cell in the cleaned table (in-place)."""
        if not self.has_cleaned or self._full_df is None:
            return
        if col not in self._full_df.columns:
            return
        try:
            # Keep as string; user can manually fix formatting.
            self._full_df.at[row_id, col] = value
        except Exception:
            return
        self._refresh_current_page()

    # ============================================================
    # HELPERS
    # ============================================================
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
        for r in page_records:
            r["__modified_flags"] = [False] * len(cols)
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
            for c in cols:
                a = orig_page.iloc[i][c]
                b = cleaned_page.iloc[i][c]
                modified = not self._safe_cell_equal(a, b)
                modified_flags_for_cols.append(modified)

            # Write flags back into the matching record by position.
            cleaned_records[i]["__modified_flags"] = modified_flags_for_cols

        # Expand flags to full column_names order if needed.
        if len(cols) != len(self.column_names):
            col_to_idx = {c: i for i, c in enumerate(cols)}
            for r in cleaned_records:
                full_flags = [False] * len(self.column_names)
                for c_i, c in enumerate(self.column_names):
                    if c in col_to_idx:
                        full_flags[c_i] = r["__modified_flags"][col_to_idx[c]]
                r["__modified_flags"] = full_flags
        else:
            # Ensure flags length matches column_names.
            for r in cleaned_records:
                if len(r["__modified_flags"]) != len(self.column_names):
                    r["__modified_flags"] = (r["__modified_flags"] + [False] * len(self.column_names))[: len(self.column_names)]
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

    @staticmethod
    def _semantic_type_to_color(semantic_type: str) -> str:
        # Hex palette (Tailwind-ish) so the frontend is independent of theme.
        mapping = {
            "BOOLEAN": "#10B981",  # emerald
            "INTEGER": "#2563EB",  # blue
            "FLOAT": "#1D4ED8",  # dark blue
            "DATETIME": "#7C3AED",  # violet
            "STRING": "#F59E0B",  # amber
            "NATURAL_LANGUAGE_TEXT": "#F59E0B",
            "DISCRETE_STRING": "#FBBF24",
            "NAMED_ENTITY": "#FBBF24",
            "DIRTY_INTEGER": "#DC2626",  # red
            "DIRTY_FLOAT": "#B91C1C",
            "DIRTY_NUMERIC": "#DC2626",
            "EMPTY": "#9CA3AF",  # gray
            "UNKNOWN": "#9CA3AF",
        }
        return mapping.get(semantic_type.upper(), "#9CA3AF")

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

        config = CleaningConfig(
            verbose=self.verbose,
            enable_validation=self.enable_validation,
            enable_validation_multi=self.enable_validation_multi,
            enable_multi_col_cleaning=self.enable_multi_col_cleaning,
            sample_sizes=self.sample_sizes,
            sample_size_validator=self.sample_size_validator,
            max_cleaning_attempts=self.max_cleaning_attempts,
            max_multi_col_attempts=self.max_multi_col_attempts,
            max_parse_attempts=self.max_parse_attempts,
            max_coding_attempts=self.max_coding_attempts,
            semaphore_limit=self.semaphore_limit,
            include_metadata=self.include_metadata,
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
            self.column_profile_rows = [
                {
                    "name": col,
                    "semantic_type": self.column_semantic_types.get(col, "UNKNOWN"),
                    "color": self.column_header_colors.get(col, "#9CA3AF"),
                }
                for col in self.column_names
            ]
            self.fd_results = fd_rows
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
            self.cleaning_summary = {}
            self.modified_cell_keys = []
            self.logs = []
            self._log_file_offset = 0
            self._log_file_path = os.path.join(rx.get_upload_dir(), "madclean_run_logs.txt")
            # Reset log file at the start of each run.
            try:
                with open(self._log_file_path, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception:
                self._log_file_path = ""
            global _CANCEL_EVENT
            _CANCEL_EVENT = threading.Event()

        config = CleaningConfig(
            verbose=self.verbose,
            enable_validation=self.enable_validation,
            enable_validation_multi=self.enable_validation_multi,
            enable_multi_col_cleaning=self.enable_multi_col_cleaning,
            sample_sizes=self.sample_sizes,
            sample_size_validator=self.sample_size_validator,
            max_cleaning_attempts=self.max_cleaning_attempts,
            max_multi_col_attempts=self.max_multi_col_attempts,
            max_parse_attempts=self.max_parse_attempts,
            max_coding_attempts=self.max_coding_attempts,
            semaphore_limit=self.semaphore_limit,
            include_metadata=self.include_metadata
        )

        llm_configs = {
            "recommender": LLM_CLIENT_MAP[self.selected_llm_key_recommender],
            "coding": LLM_CLIENT_MAP[self.selected_llm_key_coding],
            "validation": LLM_CLIENT_MAP[self.selected_llm_key_validation],
        }
        def _cancel_check():
            global _CANCEL_EVENT
            return _CANCEL_EVENT is not None and _CANCEL_EVENT.is_set()

        pipeline = Pipeline(
            llm_config=llm_configs["coding"],
            agent_llm_configs=llm_configs,
            config=config,
            log_callback=self._enqueue_log,
            cancel_check=_cancel_check,
        )
        coordinator = pipeline.cleaning_coordinator
        
        loop = asyncio.get_event_loop()
        start = time.perf_counter()
        task = loop.run_in_executor(None, pipeline.run, self._file_path)

        while not task.done():
            await asyncio.sleep(0.5)
            await self._drain_logs()
            async with self:
                self.progress_percent = int(coordinator.progress.get("current_percentage", 0))
                self.status_msg = f"Cleaning... {self.progress_percent}%"
        
        result = await task
        await self._drain_logs()
        end = time.perf_counter()
        async with self:
            if result is not None:
                cleaned_df, report = result
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
                    # Human-friendly multi-line block (matches requested format).
                    blocks: List[str] = []
                    if isinstance(self.token_usage, dict):
                        for agent, usage in self.token_usage.items():
                            if not isinstance(usage, dict):
                                continue
                            blocks.append(
                                "\n".join(
                                    [
                                        f"{str(agent).capitalize()} Agent:",
                                        f"\tInput Tokens:  {usage.get('input_tokens', 0)}",
                                        f"\tOutput Tokens: {usage.get('output_tokens', 0)}",
                                        f"\tTotal Tokens:  {usage.get('total_tokens', 0)}",
                                    ]
                                )
                            )
                    total = (report or {}).get("total_usage") or {}
                    if isinstance(total, dict) and total:
                        blocks.append(
                            "\n".join(
                                [
                                    "TOTAL:",
                                    f"\tInput Tokens:  {total.get('input_tokens', 0)}",
                                    f"\tOutput Tokens: {total.get('output_tokens', 0)}",
                                    f"\tTotal Tokens:  {total.get('total_tokens', 0)}",
                                ]
                            )
                        )
                    self.token_usage_block = "\n\n".join(blocks).strip()
                    self.runtime_seconds = float((report or {}).get("runtime_seconds", end - start))
                    self.runtime_seconds_display = f"{self.runtime_seconds:.1f}"
                    self.cleaning_summary = report or {}
                    self._update_report_code_view(self.cleaning_summary)
                    self.page_offset = 0
                    self._refresh_current_page()
                    self.status_msg = "Finished"
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
                self.total_rows = int(len(df))
                self.page_offset = 0
                self._refresh_current_page()
                self.modified_cell_keys = []
                self.token_usage = {}
                self.cleaning_summary = {}
                self.report_code_keys = []
                self.selected_report_code_key = ""
                self.generated_code_for_selected = ""
                self.selected_report_meta = {}
                # Initialize profiling UI defaults (runs after upload).
                self.column_semantic_types = {c: "UNKNOWN" for c in self.column_names}
                self.column_header_colors = {c: self._semantic_type_to_color("UNKNOWN") for c in self.column_names}
                self.column_profile_rows = [
                    {"name": c, "semantic_type": "UNKNOWN", "color": self.column_header_colors[c]}
                    for c in self.column_names
                ]
                self.profiling_status = "Waiting for profiling..."
                self.is_profiling = False
                self.status_msg = f"Loaded {file.filename}"
            # Kick off profiling in background (don’t block upload).
            # Reflex requires using `yield State.run_profiling_process` for background tasks.
            yield State.run_profiling_process
        except Exception as e:
            async with self:
                self.status_msg = f"Upload Error: {str(e)}"

    def download_cleaned_file(self):
        if self._full_df is not None:
            return rx.download(data=self._full_df.to_csv(index=False), filename=f"cleaned_{self.file_name}")