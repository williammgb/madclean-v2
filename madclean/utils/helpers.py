import hashlib
import random
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Any, Tuple

# "{{" and "}}" escapes, or a {name} / {name:spec} placeholder. Any other brace text is literal.
_PROMPT_TOKEN_RE = re.compile(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)(?::([^{}]*))?\}")


def format_prompt_template(template: str, **kwargs: Any) -> str:
    """
    Fill a prompt template the way str.format does, except that brace text which is not a
    placeholder (e.g. an example like {'k': 1}) is kept literally instead of raising.
    Injected values are inserted as-is and never parsed again.
    """
    def _replace(match: re.Match) -> str:
        token = match.group(0)
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        return format(kwargs[match.group(1)], match.group(2) or "")

    return _PROMPT_TOKEN_RE.sub(_replace, template)


def _seed_digest(seed: int, keys: tuple) -> int:
    text = "|".join([str(seed), *(str(key) for key in keys)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def seeded_random(seed: int | None, *keys: Any) -> random.Random | None:
    """
    A random.Random for one sampling purpose (e.g. seed, "validation", column, attempt), or None when no
    seed is set. Separate streams per key keep samples stable although columns are cleaned concurrently.
    """
    if seed is None:
        return None
    return random.Random(_seed_digest(seed, keys))


def seeded_generator(seed: int | None, *keys: Any) -> np.random.Generator | None:
    """numpy counterpart of seeded_random."""
    if seed is None:
        return None
    return np.random.default_rng(_seed_digest(seed, keys))

def _leading_zero_columns(raw: pd.DataFrame) -> list:
    """Columns whose non-empty values are all digits and at least one starts with a zero (ZIP codes, IDs)."""
    columns = []
    for col in raw.columns:
        values = raw[col].dropna().astype(str).str.strip()
        values = values[values != ""]
        if not values.empty and values.str.fullmatch(r"\d+").all() and values.str.match(r"0\d").any():
            columns.append(col)
    return columns


def load_dataset(file_path: Union[str, Path], keep_raw_text: bool = False) -> Optional[pd.DataFrame]:
    """
    Load a dataset from CSV, JSON or XLSX into a pandas DataFrame.
    keep_raw_text=True (CSV and XLSX) keeps placeholder text such as "N/A" as text instead of NaN, so only
    empty cells are missing, and reads digit-only columns with a leading zero as text, so the zeros stay.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        msg = f"Error: File does not exist at path: {file_path}"
        raise FileNotFoundError(msg)
    raw_text = {"keep_default_na": False, "na_values": [""]} if keep_raw_text else {}
    loaders = {
        ".csv": lambda f, **kw: pd.read_csv(f, encoding="utf-8", on_bad_lines="skip", **raw_text, **kw),
        ".json": lambda f, **kw: pd.read_json(f, encoding="utf-8"),
        ".xlsx": lambda f, **kw: pd.read_excel(f, **raw_text, **kw) # .xls?
    }
    ext = file_path.suffix.lower()
    if ext not in loaders:
        msg = f"Unsupported file type: '{ext}'. Supported types are CSV, JSON, XLSX."
        raise ValueError(msg)
    try:
        if keep_raw_text and ext in (".csv", ".xlsx"):
            text_columns = _leading_zero_columns(loaders[ext](file_path, dtype=str))
            if text_columns:
                return loaders[ext](file_path, dtype={col: str for col in text_columns})
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
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    cleaned_file_name = f"{base}_cleaned{ext}"
    cleaned_file_path = cleaned_dir / cleaned_file_name
    counter = 2
    while cleaned_file_path.exists():
        cleaned_file_name = f"{base}_cleaned_{counter}{ext}"
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


def align_dirty_cleaned_series(
    dirty_series: pd.Series, cleaned_series: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """Pair dirty and cleaned columns on the same row index for comparisons.

    Coder output often uses a default RangeIndex while the dataframe keeps the
    original index; align by position when lengths match, else intersect indices.
    """
    if cleaned_series.index.equals(dirty_series.index):
        return dirty_series, cleaned_series
    if len(cleaned_series) == len(dirty_series):
        cleaned_aligned = pd.Series(
            cleaned_series.to_numpy(copy=True),
            index=dirty_series.index,
            name=cleaned_series.name,
        )
        return dirty_series, cleaned_aligned
    common = dirty_series.index.intersection(cleaned_series.index)
    return dirty_series.loc[common], cleaned_series.loc[common]


def align_dirty_cleaned_dataframe(
    dirty_df: pd.DataFrame, cleaned_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Same as align_dirty_cleaned_series for multi-column validation samples."""
    if cleaned_df.index.equals(dirty_df.index):
        return dirty_df, cleaned_df
    if len(cleaned_df) == len(dirty_df):
        cleaned_aligned = pd.DataFrame(
            cleaned_df.to_numpy(copy=True),
            index=dirty_df.index,
            columns=cleaned_df.columns,
        )
        return dirty_df, cleaned_aligned
    common = dirty_df.index.intersection(cleaned_df.index)
    return dirty_df.loc[common], cleaned_df.loc[common]


def llm_sampling_kwargs_from_config(config: Any) -> dict[str, float]:
    """Map CleaningConfig llm_temperature / llm_top_p to kwargs for LLM clients (empty if unset)."""
    out: dict[str, float] = {}
    t = getattr(config, "llm_temperature", None)
    p = getattr(config, "llm_top_p", None)
    if t is not None:
        try:
            out["temperature"] = float(t)
        except (TypeError, ValueError):
            pass
    if p is not None:
        try:
            out["top_p"] = float(p)
        except (TypeError, ValueError):
            pass
    return out
