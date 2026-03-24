import reflex as rx
import pandas as pd
import asyncio
import os
from typing import Optional, List, Dict, Any
from madclean.config.settings import CleaningConfig
from madclean.llm.llm_registry import LLM_CLIENT_MAP
from madclean.pipeline import Pipeline

class State(rx.State):
    """Bridge between MADClean system and the UI."""
    _default_config = CleaningConfig()

    # Configuration State
    verbose: bool = _default_config.verbose
    print(verbose)
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
    
    # LLM Settings
    llm_options: List[str] = list(LLM_CLIENT_MAP.keys())
    selected_llm_key: str = llm_options[0] if llm_options else ""

    # Data State
    file_name: str = "No file selected"
    _file_path: str = "" 
    df_preview: List[Dict[str, Any]] = []
    column_names: List[str] = []
    _full_df: Optional[pd.DataFrame] = None
    token_usage: Dict[str, Any] = {}

    # ============================================================
    # EXPLICIT SETTERS
    # ============================================================
    def set_verbose(self, val: bool): self.verbose = val
    def set_enable_validation(self, val: bool): self.enable_validation = val
    def set_enable_validation_multi(self, val: bool): self.enable_validation_multi = val
    def set_enable_multi_col_cleaning(self, val: bool): self.enable_multi_col_cleaning = val
    def set_include_metadata(self, val: bool): self.include_metadata = val
    def set_selected_llm_key(self, val: str): self.selected_llm_key = val

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

    @rx.event(background=True)
    async def run_cleaning_process(self):
        if not self._file_path:
            async with self: self.status_msg = "Error: No file selected."
            return
        
        async with self:
            self.is_cleaning = True
            self.status_msg = "Initializing Pipeline..."

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

        llm_config = LLM_CLIENT_MAP[self.selected_llm_key]
        pipeline = Pipeline(llm_config=llm_config, config=config)
        coordinator = pipeline.cleaning_coordinator
        
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, pipeline.run, self._file_path)

        while not task.done():
            await asyncio.sleep(0.5)
            async with self:
                self.progress_percent = int(coordinator.progress.get("current_percentage", 0))
                self.status_msg = f"Cleaning... {self.progress_percent}%"
        
        result = await task
        async with self:
            if result is not None:
                self._full_df, report = result
                self.df_preview = self._full_df.head(100).to_dict("records")
                self.token_usage = report.get("token_usage", {})
                self.status_msg = "Finished"
            else:
                self.status_msg = "Pipeline Error."
            self.is_cleaning = False

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
                self._full_df = df
                self._file_path = outfile
                self.file_name = file.filename
                self.column_names = list(df.columns)
                self.df_preview = df.head(100).to_dict("records")
                self.status_msg = f"Loaded {file.filename}"
        except Exception as e:
            async with self: self.status_msg = f"Upload Error: {str(e)}"

    def download_cleaned_file(self):
        if self._full_df is not None:
            return rx.download(data=self._full_df.to_csv(index=False), filename=f"cleaned_{self.file_name}")