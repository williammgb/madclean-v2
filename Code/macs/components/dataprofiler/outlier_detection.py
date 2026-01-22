import numpy as np
import pandas as pd
# Local imports
from macs.components.domain.schema import OutlierResult

class OutlierDetection:
    def __init__(self, verbose: bool = False):
        """
        Class for detecting outliers using MAD and gathering context rows to assist LLM in making better decisions.
        This class implement the SingleColumnCleaner protocol.
        """
        self.verbose = verbose
    
    def analyse(self, df: pd.DataFrame, col: str, column_type: str) -> dict | None:
        if column_type not in ['INTEGER', 'FLOAT']:
            return None
        outlier_data = self._detect_outliers(df, col)
        return {"outlier_data": outlier_data} if outlier_data else None
        
    def _detect_outliers(self, df: pd.DataFrame, col: str, m_threshold: float = 5.0) -> OutlierResult | None:
        """Identifies outliers in numerical columns using the Modified Z-Score (MAD method) which is more robust to outliers."""
        def truncate(val):
            """Helper function to truncate cells with long string values to reduce token usage and minimize noise."""
            if isinstance(val, str) and len(val) > 50:
                return val[:50] + "..."
            return val
        # 1. Filter column for pure numeric values
        column = df[col]
        numeric_series = pd.to_numeric(column, errors='coerce').dropna()
        if numeric_series.empty or len(numeric_series) < 10:
            return None
        # 2. Calculate MAD score
        median = numeric_series.median()
        mad = np.median(np.abs(numeric_series - median))    
        if mad == 0:
            return None
        modified_z_score = 0.6745 * (numeric_series - median) / mad
        is_outlier = np.abs(modified_z_score) > m_threshold
        outlier_values = numeric_series[is_outlier]
        if outlier_values.empty:
            return None
        # if self.verbose: print(f"[{col}] (median {median}) has {len(outlier_values)} possible outliers: {outlier_values.values}")
        outlier_counts = outlier_values.value_counts()
        outliers_list = list(outlier_counts.items())
        # 3. Add full rows belonging to the outliers for additional context for LLM
        context_rows = []
        context_rows.append(list(df.columns))
        unique_outliers = outlier_values.unique()
        for outlier in unique_outliers:
            ids = outlier_values[outlier_values == outlier].index
            sample_ids = ids[:2]
            rows = df.loc[sample_ids].values.tolist()
            final_rows = [[truncate(val) for val in row] for row in rows]
            context_rows.extend(final_rows)
        return OutlierResult(
            outliers=outliers_list,
            median=median,
            mad=mad,
            context=context_rows
        )

####### TEST CODE #######
if __name__ == "__main__":
    ## TEST 1
    data = {
        "category": ["A", "A", "A", "B", "B", "B", "Error", "Typo", "C", "B", "A", "A"],
        "values":   [1,   2,   2,   3,   3,   3,   100,     105,    110, 3,   2,   1]
    }
    df = pd.DataFrame(data)
    detector = OutlierDetection(verbose=True)
    result = detector.analyse(df, "values", "INTEGER")
    print(result) 
    print()
    ## TEST 2
    from pathlib import Path
    from macs.utils.helpers import load_dataset
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    file_path = BASE_DIR / "data" / "benchmark_datasets" / "beers_dirty.csv"
    df = load_dataset(file_path)
    result = detector.analyse(df, "ibu", "INTEGER")
    print(result)

    # python -m macs.components.dataprofiler.outlier_detection