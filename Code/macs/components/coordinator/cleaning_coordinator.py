import asyncio
import pandas as pd
# Local imports
from macs.components.multi_agent_cleaner.multi_agent_cleaning import MultiAgentCleaning
from macs.components.dataprofiler.dataprofiler import MultiColumnCleaner
from macs.components.domain.schema import MultiColumnTask, ColumnProfile
from macs.components.coordinator.task_scheduler import TaskScheduler

class CleaningCoordinator:
    """Coordinates the Multi-Agent Cleaning Loop: determines order of asynchronous operations based on dependencies, runs all loop operations. """
    def __init__(self, 
                 multi_agent_loop: MultiAgentCleaning, 
                 multi_column_cleaners: list[MultiColumnCleaner],
                 verbose: bool = False, 
                 semaphore_limit: int = 15):
        self.verbose = verbose
        self.multi_agent_loop = multi_agent_loop
        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self.task_scheduler = TaskScheduler(verbose=verbose)
        self.multi_column_cleaners: dict[str, MultiColumnCleaner] = {}
        if multi_column_cleaners:
            for cleaner in multi_column_cleaners:
                self.multi_column_cleaners[cleaner.task_type] = cleaner
        
    def clean_dataset(self, 
                    df: pd.DataFrame, 
                    column_profiles: dict[str, ColumnProfile],
                    multi_col_tasks: list[MultiColumnTask]) -> tuple[pd.DataFrame, dict]: # change fds
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
                                multi_col_tasks: list[MultiColumnTask]) -> tuple[pd.DataFrame, dict]:
        """Actual cleaning: makes sure all tasks are executed in correct order.""" 
        self.multi_agent_loop.reset_token_usage()
        # if self.verbose: print("Cleaning dataset...")
        df_cleaned = df.copy()
        task_levels = []
        # 1. If there are constraints, determine the execution order that respects dependencies
        if multi_col_tasks:
            task_levels = self.task_scheduler.get_execution_order(multi_col_tasks)
        # 2. Run single-column operations
        column_tasks_map = {}
        isolated_col_tasks = []
        successfully_applied_cols = set()
        for col, profile in column_profiles.items():
            column_type = profile.semantic_type
            if column_type in ("EMPTY", "UNKNOWN"):
                if column_type == "EMPTY":
                    df_cleaned[col] = pd.Series([None] * len(df_cleaned), dtype='object')
                continue   
            loop_coro = self.multi_agent_loop._run_column_cleaning_async(df_cleaned, col, profile)
            task = asyncio.create_task(self._run_with_semaphore(loop_coro)) 
            column_tasks_map[col] = task
            isolated_col_tasks.append(task)
        # 3. Run multi-column operations
        for level in task_levels:
            current_tasks = []
            for task_info in level:
                task_coro = asyncio.create_task(self._multi_col_task_wrapper(
                        df_cleaned, 
                        task_info, 
                        column_tasks_map, 
                        successfully_applied_cols))
                current_tasks.append(task_coro)
            # Wait for all level's tasks to finish before updating DataFrame
            results = await asyncio.gather(*current_tasks)
            for result in results:
                _, final_cleaned_columns = result
                if isinstance(final_cleaned_columns, pd.DataFrame):
                    cols_to_update = final_cleaned_columns.columns
                    df_cleaned[cols_to_update] = final_cleaned_columns
        # 4. Wait for remaining single-column operations and update columns
        final_isolated_results = await asyncio.gather(*isolated_col_tasks)
        for col, final_cleaned_column in final_isolated_results:
            if final_cleaned_column is not None and col not in successfully_applied_cols:
                df_cleaned[col] = final_cleaned_column
                successfully_applied_cols.add(col)
        return df_cleaned, self.multi_agent_loop.token_usage

    async def _run_with_semaphore(self, coro):
        async with self.semaphore:
            return await coro

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
            # if self.verbose: print(f"No cleaner for task type {task_type}. Skipping task...")
            return target_columns, None
        updated_task_data = cleaner.get_data(df, task_info) 
        if not updated_task_data:
            return target_columns, None # No issues found
        async with self.semaphore:       
            return await self.multi_agent_loop._run_multi_col_cleaning_async(df, updated_task_data)
    
    