import pandas as pd

class AdditionalComponent:
    """
    Demonstrates how to extend the DataProfiler by adding custom components.
    This class implement the SingleColumnCleaner protocol.
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def analyse(self, df: pd.DataFrame, col: str, column_type: str) -> dict:
        if column_type in ['INTEGER', 'FLOAT']:
            series = df[col]
            numeric_series = pd.to_numeric(series, errors='coerce')
            data =  {
                "mean": numeric_series.mean(),
                "min": numeric_series.min(),
                "max": numeric_series.max()
            }
            return {"additional_data": data}    
    

