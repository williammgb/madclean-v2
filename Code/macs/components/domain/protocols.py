from typing import Protocol
import pandas as pd
# local imports
from macs.components.domain.schema import MultiColumnTask

class SingleColumnCleaner(Protocol):
    """Demonstrates how a single column cleaning component should look like."""
    def analyse(self, df: pd.DataFrame, col: str, column_type: str) -> dict:
        """
        Returns a dict of data for the column data profile.
        Key value must equal variable in ColumnProfile
        Example: {'outlier_data: OutlierResult(...)}
        """
        pass

class MultiColumnCleaner(Protocol):
    """Demonstrates how a multi-column cleaning component should look like."""
    @property
    def task_type(self) -> str:
        """Returns the type string."""
        pass
    def detect(self, df: pd.DataFrame) -> list[MultiColumnTask]:
        """Initial scan of violations etc., returns list of task dictionaries"""
        pass
    def get_data(self, df: pd.DataFrame, task_info: dict) -> MultiColumnTask | None:
        """Second scan, for updated (partly-cleaned) df. Only for specific task."""
        pass  