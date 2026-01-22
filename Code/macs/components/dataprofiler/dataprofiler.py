import pandas as pd
# Local imports
from macs.components.dataprofiler.semantic_mapping import SemanticTypeDetection
from macs.components.dataprofiler.data_selection import DataSampler
from macs.components.domain.schema import ColumnProfile, MultiColumnTask
from macs.components.domain.protocols import SingleColumnCleaner, MultiColumnCleaner

class DataProfiler:
    """Class that combines all dataset profiling components"""
    def __init__(self, 
                 single_col_cleaners: list[SingleColumnCleaner] = None,
                 multi_col_cleaners: list[MultiColumnCleaner] = None,
                 verbose: bool = False):
        self.verbose = verbose
        self.type_detector = SemanticTypeDetection()
        self.data_sampler = DataSampler() 
        self.single_col_cleaners = single_col_cleaners if single_col_cleaners is not None else []
        self.multi_col_cleaners = multi_col_cleaners if multi_col_cleaners is not None else []
         
    def analyse(self, df: pd.DataFrame) -> tuple[dict[str, ColumnProfile], list[MultiColumnTask]]:
        """Runs all data profiling components such as detecting semantic types, outlier detection, etc."""
        # if self.verbose: print("Profiling data...")    
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
            multi_col_tasks.extend(tasks)
        return profiles, multi_col_tasks

####### TEST CODE #######       
if __name__ == "__main__":
    from pathlib import Path
    from macs.utils.helpers import load_dataset
    from macs.components.dataprofiler.outlier_detection import OutlierDetection
    from macs.components.dataprofiler.functional_dependencies import FunctionalDependencies
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    file_path = BASE_DIR / "data" / "benchmark_datasets" / "beers_dirty.csv"
    df = load_dataset(file_path)
    profiler = DataProfiler(
        single_col_cleaners=[OutlierDetection(verbose=False)],
        multi_col_cleaners=[FunctionalDependencies(verbose=False)],
        verbose=True,
        )
    profiles, multi_col_tasks = profiler.analyse(df)
    print(profiles['ibu'])
    print()
    print(multi_col_tasks[0])

    # python -m macs.components.dataprofiler.dataprofiler


