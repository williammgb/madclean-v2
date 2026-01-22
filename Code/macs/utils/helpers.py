import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Any

def load_dataset(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load a dataset from CSV, JSON or XLSX into a pandas DataFrame."""
    file_path = Path(file_path)
    if not file_path.exists():
        msg = f"Error: File does not exist at path: {file_path}"
        raise FileNotFoundError(msg)
    loaders = {
        ".csv": lambda f: pd.read_csv(f, encoding="utf-8", on_bad_lines="skip"),
        ".json": lambda f: pd.read_json(f, encoding="utf-8"),
        ".xlsx": lambda f: pd.read_excel(f) # .xls?
    }
    ext = file_path.suffix.lower()
    if ext not in loaders:
        msg = f"Unsupported file type: '{ext}'. Supported types are CSV, JSON, XLSX."
        raise ValueError(msg)
    try:
        return loaders[ext](file_path)
    except Exception as e:
        msg = f"An error occurred while reading the file: {e}"
        raise

def save_dataset(cleaned_df: pd.DataFrame, file_path: str, base_dir: Path):
    """Save the cleaned dataset in the same format as the input dirty dataset."""
    input_path = Path(file_path)
    ext = input_path.suffix.lower()
    base = input_path.stem
    if base.endswith("_dirty"):
        base = base[:-6]
    cleaned_dir = base_dir / "data" / "cleaned"
    cleaned_file_name = f"{base}_cleaned{ext}"
    cleaned_file_path = cleaned_dir / cleaned_file_name
    counter = 2
    while cleaned_file_path.exists():
        cleaned_file_name = f"{base}_cleaned_{counter}.csv"
        cleaned_file_path = cleaned_dir / cleaned_file_name
        counter += 1 
    if ext == ".csv":
        cleaned_df.to_csv(cleaned_file_path, index=False)
    elif ext == ".xlsx":
        cleaned_df.to_excel(cleaned_file_path, index=False)
    elif ext == ".json":
        cleaned_df.to_json(cleaned_file_path, orient="records", indent=2)
    return cleaned_file_path

def subsample_dataframe(df: pd.DataFrame, max_cells_threshold: int = 50000,
                        min_sample_size: int = 100, max_sample_size: int = 1000,
                        verbose: bool = False) -> pd.DataFrame:
    """Subsamples the DataFrame if it exceeds a cell count threshold."""
    if df.empty:
        return df
    rows, cols = df.shape
    total_cells = rows * cols
    if total_cells <= max_cells_threshold and rows <= 1.5 * max_sample_size:
        # if verbose: print(f"DataFrame has {rows} rows, {cols} columns. Analyzing the full dataset.")
        return df
    max_allowed_rows = max_cells_threshold // cols
    sample_size = min(rows, max(min_sample_size, min(max_sample_size, max_allowed_rows)))
    # if verbose: print(f"DataFrame has {total_cells} cells, exceeding threshold. Sampling {sample_size} rows.")
    df_no_empty = df.dropna(how='all')
    df_to_sample = df_no_empty if len(df_no_empty) >= sample_size else df
    return df_to_sample.sample(n=sample_size, random_state=42)

def format_list_for_prompt(values: List[Any]) -> List[str]:
    if not values:
        return []
    formatted_values = []
    for v in values:
        # converting numpy scalars to python scalars
        if hasattr(v, "item") and not isinstance(v, (list, tuple, set, np.ndarray, pd.Series)):
            v = v.item()
        # dealing with non-string columns, converting to string
        if isinstance(v, (list, set, tuple, np.ndarray, pd.Series)):
            formatted_values.append(f'"{v}"')
        elif pd.isna(v):
            formatted_values.append('NaN')
        elif isinstance(v, str):
            formatted_values.append(f'"{v}"')
        else:
            formatted_values.append(str(v))
    return formatted_values
        