import pandas as pd
from pathlib import Path
import time
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
    def __init__(self, llm_config: dict, config: CleaningConfig = None):
        self.config = config or CleaningConfig() # use default if none is provided
        self.verbose = self.config.verbose
        # 1. Initialise LLM client
        client_class = llm_config["class"]
        if client_class == OpenAIClient:
            self.client = OpenAIClient(
                model_name=llm_config["default_model"],
                base_url=llm_config.get("base_url")
            )
        else:
            self.client = client_class(model_name=llm_config["default_model"])
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
        self.multi_agent_loop = MultiAgentCleaning(
            llm_client=self.client,
            llm_role=llm_config["role"],
            config=self.config
        )
        # 5. Initialise Coordinator
        self.cleaning_coordinator = CleaningCoordinator(
            self.multi_agent_loop, 
            multi_column_cleaners=self.multi_col_cleaners,
            config=self.config
        )
        if self.verbose: 
            print("=" * 35)
            print(f"Pipeline initialized with LLM: {llm_config['default_model']}")
    
    @staticmethod
    def get_total_usage(token_usage):
        return {
            "input_tokens": sum(a["input_tokens"] for a in token_usage.values()),
            "output_tokens": sum(a["output_tokens"] for a in token_usage.values()),
            "total_tokens": sum(a["total_tokens"] for a in token_usage.values()),
        }
    
    def run(self, file_path: str, save_cleaned: bool = False) -> tuple[pd.DataFrame | None, dict | None]:
        if self.verbose: 
            print(f"Pipeline started for file: {file_path}")
        start_time = time.perf_counter() 
        # 1. Load file into DataFrame, return None if failed or empty
        try:
            dirty_df = load_dataset(file_path)
        except (FileNotFoundError, ValueError, Exception) as e:
            if self.verbose:
                print(f"Loading Failed for {file_path}. {type(e).__name__}: {e}")
                print("Pipeline terminated for this file.")
                print("=" * 35)
            return None, None
        if dirty_df.empty:
            if self.verbose: 
                print("Dataset is empty. Pipeline terminated.")
                print("=" * 35)
            return None, None
        # 2. Run DataProfiler
        if self.verbose: print("Profiling data...") 
        profiles, multi_col_tasks = self.data_profiler.analyse(dirty_df) 
        if not profiles:
            if self.verbose:
                print("Pipeline terminated because dataset could not be profiled.")
                print("=" * 35) 
            return None, None
        # 3. Clean DataFrame
        if self.verbose: print("Starting asynchronous LLM cleaning...")
        cleaned_df, cleaning_report = self.cleaning_coordinator.clean_dataset(
            dirty_df,
            column_profiles=profiles,
            multi_col_tasks=multi_col_tasks
        )
        end_time = time.perf_counter()
        runtime = (end_time - start_time)
        # 4. Print runtime and token usage --> UPDATE WITH AGENTS
        if self.verbose:
            token_usage = cleaning_report["token_usage"]
            print("-" * 35)
            print(f"Runtime: {runtime}s")
            print(f"Token usage for {file_path}:")
            for agent, usage in token_usage.items():
                print(f"\n  {agent.capitalize()} agent:")
                print(f"      Input tokens:   {usage['input_tokens']:>10,}")
                print(f"      Output tokens:  {usage['output_tokens']:>10,}")
                print(f"      Total tokens:   {usage['total_tokens']:>10,}")
            total_usage = self.get_total_usage(token_usage)
            print("\n  TOTAL:")
            print(f"      Input tokens:   {total_usage['input_tokens']:>10,}")
            print(f"      Output tokens:  {total_usage['output_tokens']:>10,}")
            print(f"      Total tokens:   {total_usage['total_tokens']:>10,}")
            print("-" * 35)
        # 5. Save cleaned DataFrame
        if save_cleaned:
            base_dir = Path(__file__).resolve().parent.parent # Framework root directory
            cleaned_file_path = save_dataset(cleaned_df, file_path, base_dir)
            if self.verbose: print(f"Saved cleaned DataFrame to {cleaned_file_path}")
        if self.verbose:
            print(f"Pipeline finished for file: {file_path}")
            print("=" * 35)
        return cleaned_df, cleaning_report


    