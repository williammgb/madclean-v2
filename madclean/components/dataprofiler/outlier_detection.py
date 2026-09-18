import numpy as np
import pandas as pd
# Local imports
from madclean.components.domain.schema import OutlierResult

class OutlierDetection:
    """
    Class for detecting outliers using MAD and gathering context rows to assist LLM in making better decisions.
    This class implement the SingleColumnCleaner protocol.
    """
    def __init__(self):
        pass
    
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