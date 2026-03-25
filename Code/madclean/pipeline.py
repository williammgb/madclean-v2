import pandas as pd
from pathlib import Path
import time
import asyncio
from dotenv import load_dotenv
load_dotenv()
# Local imports
from madclean.components.dataprofiler.dataprofiler import DataProfiler
from madclean.components.dataprofiler.outlier_detection import OutlierDetection
from madclean.components.dataprofiler.functional_dependencies import FunctionalDependencies
from madclean.components.multi_agent_cleaner.multi_agent_cleaning import MultiAgentCleaning
from madclean.components.coordinator.cleaning_coordinator import CleaningCoordinator
from madclean.utils.helpers import load_dataset, save_dataset
from madclean.llm.llm_clients import OpenAIClient
from madclean.config.settings import CleaningConfig

class Pipeline:
    def __init__(
        self,
        llm_config: dict,
        agent_llm_configs: dict | None = None,
        config: CleaningConfig = None,
        log_callback=None,
        verbose: bool | None = None,
        cancel_check=None,
    ):
        self.config = config or CleaningConfig() # use default if none is provided
        if verbose is not None:
            self.config.verbose = verbose
        self.verbose = self.config.verbose
        self._log_callback = log_callback
        self._cancel_check = cancel_check

        def _log(msg: str):
            if not self.verbose:
                return
            if callable(self._log_callback):
                try:
                    self._log_callback(msg)
                    return
                except Exception:
                    pass
            print(msg)

        self._log = _log
        def _create_client(cfg: dict):
            client_class = cfg["class"]
            if client_class == OpenAIClient:
                return OpenAIClient(
                    model_name=cfg["default_model"],
                    base_url=cfg.get("base_url"),
                )
            return client_class(model_name=cfg["default_model"])

        self.agent_llm_configs = agent_llm_configs
        # Default/single client (backwards compatible).
        self.client = _create_client(llm_config)
        # 2. Initialise single column cleaning components
        self.single_col_cleaners = [
            OutlierDetection()
        ]
        self.multi_col_cleaners = [
            FunctionalDependencies()
        ]
        # 3. Initialise DataProfiler
        self.data_profiler = DataProfiler(
            single_col_cleaners=self.single_col_cleaners,
            multi_col_cleaners=self.multi_col_cleaners,
            config=self.config 
        )
        # 4. Initialise LLM Agents
        if self.agent_llm_configs:
            agent_specs = {
                k: {
                    "client": _create_client(v),
                    "role": v["role"],
                    "model": v.get("default_model"),
                }
                for k, v in self.agent_llm_configs.items()
            }
            self.multi_agent_loop = MultiAgentCleaning(agent_specs=agent_specs, config=self.config)
        else:
            self.multi_agent_loop = MultiAgentCleaning(
                llm_client=self.client,
                llm_role=llm_config["role"],
                config=self.config
            )
        if hasattr(self.multi_agent_loop, "set_cancel_check"):
            self.multi_agent_loop.set_cancel_check(self._cancel_check)
        # 5. Initialise Coordinator
        self.cleaning_coordinator = CleaningCoordinator(
            self.multi_agent_loop, 
            multi_column_cleaners=self.multi_col_cleaners,
            config=self.config,
            log_callback=self._log,
            cancel_check=self._cancel_check,
        )
        if self.verbose: 
            self._log("=" * 35)
            self._log(f"Pipeline initialized with LLM: {llm_config['default_model']}")
    
    @staticmethod
    def get_total_usage(token_usage):
        return {
            "input_tokens": sum(a["input_tokens"] for a in token_usage.values()),
            "output_tokens": sum(a["output_tokens"] for a in token_usage.values()),
            "total_tokens": sum(a["total_tokens"] for a in token_usage.values()),
        }
    
    def run(self, file_path: str, save_cleaned: bool = False) -> tuple[pd.DataFrame | None, dict | None]:
        import sys
        old_stdout = None
        stream_to_log = None
        if callable(self._log_callback) and self.verbose:
            old_stdout = sys.stdout

            class _StreamToLog:
                def __init__(self, cb):
                    self._cb = cb
                    self._buffer = ""

                def write(self, s):
                    if not s:
                        return 0
                    self._buffer += s
                    while "\n" in self._buffer:
                        line, self._buffer = self._buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            try:
                                self._cb(line)
                            except Exception:
                                pass
                    return len(s)

                def flush(self):
                    return None

            stream_to_log = _StreamToLog(self._log_callback)
            sys.stdout = stream_to_log

        if self.verbose: 
            self._log(f"Pipeline started for file: {file_path}")
        start_time = time.perf_counter() 
        try:
            # 1. Load file into DataFrame, return None if failed or empty
            try:
                dirty_df = load_dataset(file_path)
            except (FileNotFoundError, ValueError, Exception) as e:
                if self.verbose:
                    self._log(f"Loading Failed for {file_path}. {type(e).__name__}: {e}")
                    self._log("Pipeline terminated for this file.")
                    self._log("=" * 35)
                return None, None
            if dirty_df.empty:
                if self.verbose:
                    self._log("Dataset is empty. Pipeline terminated.")
                    self._log("=" * 35)
                return None, None
            # 2. Run DataProfiler
            if self.verbose:
                self._log("Profiling data...")
            profiles, multi_col_tasks = self.data_profiler.analyse(dirty_df)
            if not profiles:
                if self.verbose:
                    self._log("Pipeline terminated because dataset could not be profiled.")
                    self._log("=" * 35)
                return None, None

            # 3. Clean DataFrame
            if self.verbose:
                self._log("Starting asynchronous LLM cleaning...")
            try:
                cleaned_df, cleaning_report = self.cleaning_coordinator.clean_dataset(
                    dirty_df,
                    column_profiles=profiles,
                    multi_col_tasks=multi_col_tasks
                )
            except asyncio.CancelledError:
                # Cooperative cancellation from UI "Stop" button.
                cleaned_df = None
                cleaning_report = {
                    "cancelled": True,
                    "token_usage": getattr(self.multi_agent_loop, "token_usage", {}),
                }
            end_time = time.perf_counter()
            runtime = (end_time - start_time)
            if cleaning_report is None:
                cleaning_report = {}
            cleaning_report["runtime_seconds"] = runtime

            # 4. Print runtime and token usage --> UPDATE WITH AGENTS
            token_usage = cleaning_report.get("token_usage", {})
            if token_usage:
                cleaning_report["total_usage"] = self.get_total_usage(token_usage)

            # Only print usage details when running outside the UI.
            if self.verbose and not callable(self._log_callback):
                self._log("-" * 35)
                self._log(f"Runtime: {runtime}s")
                self._log(f"Token usage for {file_path}:")
                for agent, usage in token_usage.items():
                    self._log(f"\n  {agent.capitalize()} agent:")
                    self._log(f"      Input tokens:   {usage['input_tokens']:>10,}")
                    self._log(f"      Output tokens:  {usage['output_tokens']:>10,}")
                    self._log(f"      Total tokens:   {usage['total_tokens']:>10,}")
                total_usage = cleaning_report.get("total_usage", {})
                self._log("\n  TOTAL:")
                self._log(f"      Input tokens:   {total_usage.get('input_tokens', 0):>10,}")
                self._log(f"      Output tokens:  {total_usage.get('output_tokens', 0):>10,}")
                self._log(f"      Total tokens:   {total_usage.get('total_tokens', 0):>10,}")
                self._log("-" * 35)

            # 5. Save cleaned DataFrame
            if save_cleaned:
                base_dir = Path(__file__).resolve().parent.parent  # Framework root directory
                cleaned_file_path = save_dataset(cleaned_df, file_path, base_dir)
                if self.verbose:
                    self._log(f"Saved cleaned DataFrame to {cleaned_file_path}")
            if self.verbose:
                self._log(f"Pipeline finished for file: {file_path}")
                self._log("=" * 35)
            return cleaned_df, cleaning_report
        finally:
            if old_stdout is not None:
                sys.stdout = old_stdout


    