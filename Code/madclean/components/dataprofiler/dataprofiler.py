import pandas as pd
# Local imports
from madclean.components.dataprofiler.semantic_mapping import SemanticTypeDetection
from madclean.components.dataprofiler.data_selection import DataSampler
from madclean.components.domain.schema import ColumnProfile, MultiColumnTask
from madclean.components.domain.protocols import SingleColumnCleaner, MultiColumnCleaner
from madclean.config.settings import CleaningConfig

class DataProfiler:
    """Class that combines all dataset profiling components"""
    def __init__(self, 
                 config: CleaningConfig,
                 single_col_cleaners: list[SingleColumnCleaner] = None,
                 multi_col_cleaners: list[MultiColumnCleaner] = None):
        self.config = config
        self.type_detector = SemanticTypeDetection()
        self.data_sampler = DataSampler(sample_sizes=self.config.sample_sizes) 
        self.single_col_cleaners = single_col_cleaners if single_col_cleaners is not None else []
        self.multi_col_cleaners = multi_col_cleaners if multi_col_cleaners is not None else []
         
    def analyse(self, df: pd.DataFrame) -> tuple[dict[str, ColumnProfile], list[MultiColumnTask]]:
        """Runs all data profiling components such as detecting semantic types, outlier detection, etc."""   
        if df.empty:
            return {}, []
        # 1. Detecting semantic types
        column_types = self.type_detector.detect_types(df)
        if not column_types:
            return {}, []
        # 2. Initialize the dictionary of ColumnProfiler
        profiles: dict[str, ColumnProfile] = {}
        for col in df.columns:
            column_type = column_types.get(col, "UNKNOWN")
            # 3. Sampling data based on semantic type
            column_sample = ""
            if column_type not in ['EMPTY', "UNKNOWN"]:
                column_sample = self.data_sampler.sample_column(df[col], col, column_type)
            # 4. Create base profile object
            profile = ColumnProfile(
                name=col,
                semantic_type=column_type,
                sample=column_sample
            )
            # 5. Running single column analysis (e.g., Outlier Detection)
            for cleaner in self.single_col_cleaners:
                data = cleaner.analyse(df, col, column_type)
                if data:
                    for k, v in data.items():
                        # If profiler has assigned attribute for this cleaner, set that attribute
                        if hasattr(profile, k):
                            setattr(profile, k, v)
                        # Otherwise add it to the metadata dict
                        else:
                            profile.metadata[k] = v 
            profiles[col] = profile
        # 6. Running multi-column analysis (e.g., Functional Dependency Detection)
        multi_col_tasks: list[MultiColumnTask] = []
        for multi_col_cleaner in self.multi_col_cleaners:
            tasks = multi_col_cleaner.detect(df)
            # if self.config.verbose: print(f"{len(tasks)} {multi_col_cleaner.task_type}s detected.")
            multi_col_tasks.extend(tasks)
        return profiles, multi_col_tasks