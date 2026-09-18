import asyncio
import pandas as pd
# Local imports
from madclean.components.multi_agent_cleaner.multi_agent_cleaning import MultiAgentCleaning
from madclean.components.dataprofiler.dataprofiler import MultiColumnCleaner
from madclean.components.domain.schema import MultiColumnTask, ColumnProfile
from madclean.components.domain.report import (
    CleaningReport,
    ColumnReport,
    FDReport,
    SkippedColumnReport,
    TraceEvent,
    TraceStep,
)
from madclean.components.coordinator.task_scheduler import TaskScheduler
from madclean.config.settings import CleaningConfig

class CleaningCoordinator:
    """Coordinates the Multi-Agent Cleaning Loop: determines order of asynchronous operations based on dependencies, runs all loop operations. """
    def __init__(self, 
                 multi_agent_loop: MultiAgentCleaning, 
                 multi_column_cleaners: list[MultiColumnCleaner],
                 config: CleaningConfig,
                 log_callback=None,
                 cancel_check=None):
        self.config = config
        self.multi_agent_loop = multi_agent_loop
        self.semaphore = asyncio.Semaphore(self.config.semaphore_limit)
        self.task_scheduler = TaskScheduler()
        self.multi_column_cleaners: dict[str, MultiColumnCleaner] = {c.task_type: c for c in multi_column_cleaners}
        self._log_callback = log_callback
        self.cancel_check = cancel_check
        self.progress = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "current_percentage": 0.0,
            "status": "idle"
        }
    
    async def _tracked_col_task(self, col, profile, df_cleaned):
        try:
            col, cleaned_series, status_msg = await self.multi_agent_loop._run_column_cleaning_async(
                df_cleaned, col, profile
            )
        except Exception as exc:
            # One broken column must not end the run: report it failed and keep its original values.
            cleaned_series = None
            status_msg = self._record_task_failure(col, exc, ColumnReport(
                datatype=profile.semantic_type,
                already_clean=False,
                cleaned=False,
                attempts=0,
                generated_code="",
                cleaning_validated=False,
            ))
        self._update_progress(message=status_msg)
        return col, cleaned_series

    async def _tracked_multi_col_task(self, df_cleaned, task_info, column_tasks_map, successfully_applied_cols):
        try:
            target_cols, cleaned_df, status_msg = await self._multi_col_task_wrapper(
                df_cleaned, task_info, column_tasks_map, successfully_applied_cols
            )
        except Exception as exc:
            target_cols, cleaned_df = task_info.target_columns, None
            status_msg = self._record_task_failure(task_info.verbose_key, exc, FDReport(
                target_columns=task_info.target_columns,
                cleaned=False,
                attempts=0,
                cleaning_validated=False,
                generated_code="",
            ))
        self._update_progress(message=status_msg)
        return target_cols, cleaned_df

    def _record_task_failure(self, key: str, exc: Exception, entry: ColumnReport | FDReport) -> str:
        """Fills in why a task that raised failed, reports it, traces it, and returns its log message."""
        reason = f"{type(exc).__name__}: {exc}"
        entry.reason = reason
        entry.trace_steps = [TraceStep(id="finished", title="Finished", status="failed", output=reason)]
        self.multi_agent_loop.cleaning_report[key] = entry
        self.multi_agent_loop._emit_trace(
            TraceEvent(column=key, step_id="finished", title="Finished", status="failed", output=reason)
        )
        return f"[{key}] FAILED cleaning: {reason}"
    
    async def _run_with_semaphore(self, coro):
            async with self.semaphore:
                return await coro

    def _update_progress(self, message: str = ""):
        if self.progress["total_tasks"] > 0:
            if callable(self.cancel_check) and self.cancel_check():
                raise asyncio.CancelledError()
            self.progress["completed_tasks"] += 1
            self.progress["current_percentage"] = (
                self.progress["completed_tasks"] / self.progress["total_tasks"]
            ) * 100

            if self.config.verbose: 
                # prefix = f"[{self.progress['current_percentage']:>5.1f}%]"
                # line = f"{prefix} - {message}"
                line = message
                if callable(self._log_callback):
                    try:
                        self._log_callback(line)
                    except Exception:
                        # Never crash cleaning due to UI logging.
                        print(line)
                else:
                    print(line)

    def clean_dataset(self, 
                    df: pd.DataFrame, 
                    column_profiles: dict[str, ColumnProfile],
                    multi_col_tasks: list[MultiColumnTask]) -> tuple[pd.DataFrame, CleaningReport]: # change fds
        """Function to run cleaning synchronous via pipeline."""
        try:
            # for normal scripts
            return asyncio.run(self.clean_dataset_async(df, column_profiles, multi_col_tasks))
        except RuntimeError as e:
            # for environments with a running event loop (like Jupyter)
            if "running event loop" in str(e):
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                coro = self.clean_dataset_async(df, column_profiles, multi_col_tasks) 
                return loop.run_until_complete(coro)
            else:
                raise e

    async def clean_dataset_async(self, 
                                df: pd.DataFrame, 
                                column_profiles: dict[str, ColumnProfile],
                                multi_col_tasks: list[MultiColumnTask]) -> tuple[pd.DataFrame, CleaningReport]:
        """Actual cleaning: makes sure all tasks are executed in correct order."""
        df_cleaned = df.copy()
        try:
            self.multi_agent_loop.reset_token_usage()
            skip_cols = set(getattr(self.config, "skip_columns", []) or [])

            # 1. Calculate total tasks
            col_count = len(column_profiles)
            multi_col_count = len(multi_col_tasks) if self.config.enable_multi_col_cleaning else 0
            self.progress.update({
                "total_tasks": col_count + multi_col_count,
                "completed_tasks": 0, "current_percentage": 0.0, "status": "running"
            })
            # 3. Run single-column operations
            column_tasks_map = {}
            isolated_col_tasks = []
            successfully_applied_cols = set()
            for col, profile in column_profiles.items():
                if callable(self.cancel_check) and self.cancel_check():
                    raise asyncio.CancelledError()
                if col in skip_cols:
                    # Respect GUI: user marked this column as already clean.
                    skipped_output = "User has determined this column is already clean."
                    self.multi_agent_loop.cleaning_report[col] = ColumnReport(
                        datatype=profile.semantic_type,
                        already_clean=True,
                        cleaned=False,
                        attempts=0,
                        generated_code="",
                        cleaning_validated=False,
                        trace_steps=[
                            TraceStep(id="finished", title="Finished", status="completed", output=skipped_output)
                        ],
                    )
                    # Emit a trace event so the GUI pipeline view shows a Finished card.
                    try:
                        self.multi_agent_loop._emit_trace(
                            TraceEvent(
                                column=col,
                                step_id="finished",
                                title="Finished",
                                status="completed",
                                output=skipped_output,
                            )
                        )
                    except Exception:
                        # Tracing must never break cleaning.
                        pass
                    msg = f"[{col}] Skipping: user marked this column as already clean."
                    self._update_progress(message=msg)
                    continue
                if profile.semantic_type in ("EMPTY", "UNKNOWN"):
                    self.multi_agent_loop.cleaning_report[col] = SkippedColumnReport(
                        datatype=profile.semantic_type,
                        already_clean=False,
                        cleaned=False,
                        attempts=0,
                        cleaning_validated=False,
                        reason="Empty column or unknown data type",
                    )
                    msg = f"[{col}] Skipping empty column or unknown data type"
                    self._update_progress(message=msg) 
                    continue   
                task = asyncio.create_task(
                    self._run_with_semaphore(self._tracked_col_task(col, profile, df_cleaned))
                ) 
                column_tasks_map[col] = task
                isolated_col_tasks.append(task)
            # 4. Run multi-column operations
            task_levels = []
            if self.config.enable_multi_col_cleaning and multi_col_tasks:
                task_levels = self.task_scheduler.get_execution_order(multi_col_tasks)

            for level in task_levels:
                if callable(self.cancel_check) and self.cancel_check():
                    raise asyncio.CancelledError()
                current_tasks = []
                for task_info in level:
                    task_coro = asyncio.create_task(
                        self._tracked_multi_col_task(
                            df_cleaned, task_info, column_tasks_map, successfully_applied_cols
                        )
                    )
                    current_tasks.append(task_coro)
                # Wait for all level's tasks to finish before updating DataFrame
                results = await asyncio.gather(*current_tasks)
                for result in results:
                    _, final_cleaned_columns = result
                    if isinstance(final_cleaned_columns, pd.DataFrame):
                        df_cleaned[final_cleaned_columns.columns] = final_cleaned_columns
                    # add else: when no 
            # 5. Wait for remaining single-column operations and update columns
            final_isolated_results = await asyncio.gather(*isolated_col_tasks)
            for col, final_cleaned_column in final_isolated_results:
                if final_cleaned_column is not None and col not in successfully_applied_cols:
                    df_cleaned[col] = final_cleaned_column
                    successfully_applied_cols.add(col)

            self.progress["status"] = "finished"
            return df_cleaned, CleaningReport(
                entries=self.multi_agent_loop.cleaning_report,
                token_usage=self.multi_agent_loop.token_usage,
                per_task_usage=self.multi_agent_loop.per_task_usage,
            )
        except asyncio.CancelledError:
            # Cooperative cancellation: return whatever progress we have so far.
            self.progress["status"] = "cancelled"
            return df_cleaned, CleaningReport(
                entries=self.multi_agent_loop.cleaning_report,
                token_usage=self.multi_agent_loop.token_usage,
                per_task_usage=self.multi_agent_loop.per_task_usage,
                cancelled=True,
            )

    async def _multi_col_task_wrapper(self, 
                                      df: pd.DataFrame, 
                                      task_info: MultiColumnTask, 
                                      column_tasks_map: dict, 
                                      successfully_applied_cols: set):
        """
        Generic wrapper for any multi-column operation.
        Makes sure applying multi-column operation is only done after the involved columns are cleaned individually
        """ 
        task_type = task_info.task_type
        target_columns = task_info.target_columns
        # 1. Wait for dependencies
        dependency_tasks = []
        for col in target_columns:
            if col in column_tasks_map:
                dependency_tasks.append(column_tasks_map[col])
        column_results = await asyncio.gather(*dependency_tasks) 
        # 2. Apply individual updates
        for col, final_cleaned_column in column_results:
            if isinstance(final_cleaned_column, pd.Series) and col not in successfully_applied_cols:
                df[col] = final_cleaned_column 
                successfully_applied_cols.add(col)
        # 3. Re-profile data after updates
        cleaner = self.multi_column_cleaners.get(task_type)
        if not cleaner:
            raise ValueError(f"Cleaner of {task_type} not configured in pipeline. First update pipeline configuration.")
        
        updated_task_data = cleaner.get_data(df, task_info) 
        if not updated_task_data:
            msg = f"[{task_info.verbose_key}] No {task_type} errors detected after cleaning of columns."
            return target_columns, None, msg
        async with self.semaphore:       
            return await self.multi_agent_loop._run_multi_col_cleaning_async(df, updated_task_data)
    
    