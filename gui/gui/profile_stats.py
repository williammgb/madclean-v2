"""Per-column profile numbers for the Profile view.

Everything here is computed on this machine from the loaded table, before any model is called. Each
column gets its counts, its most common values and value shapes, a histogram when it holds numbers,
and a list of findings: values that break a shape almost every other value follows, text in a number
column, numbers far from the rest, placeholder text such as "N/A", and extra spaces.

Every value the view shows is a string, so Reflex can render it without type juggling.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd

NUMERIC_TYPES = {"INTEGER", "FLOAT", "DIRTY_INTEGER", "DIRTY_FLOAT"}
PLACEHOLDERS = {"n/a", "na", "null", "none", "nan", "-", "--", "?", "unknown", "missing", "empty"}

# A shape that covers at least this share of a column makes the values outside it worth a look.
DOMINANT_SHAPE_PCT = 95.0
# Shapes are counted on at most this many values, so a large table stays fast.
SHAPE_SAMPLE = 5000
HIST_BINS = 16
TOP_VALUES = 5
TOP_SHAPES = 3
SHOWN_VALUES = 8
_NUMBER_IN_TEXT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def _char_token(ch: str, loose: bool) -> str:
    if ch.isdigit():
        return r"\d"
    if ch.isalpha():
        if loose:
            return "[A-Za-z]"
        return "[A-Z]" if ch.isupper() else "[a-z]"
    if ch.isspace():
        return r"\s"
    return re.escape(ch)


def value_shape(value: str, loose: bool = False) -> str:
    r"""The regex shape of one value.

    The exact shape keeps each run's length and letter case ("35233" -> ^\d{5}$). The loose shape
    drops lengths and case and folds words separated by spaces together ("fort payne" and
    "Boston" -> ^[A-Za-z ]+$), so free text with varying lengths still has one shape.
    """
    if not value:
        return "^$"
    runs: List[List[Any]] = []
    for ch in value:
        tok = _char_token(ch, loose)
        if runs and runs[-1][0] == tok:
            runs[-1][1] += 1
        else:
            runs.append([tok, 1])
    if loose:
        body = "".join(
            tok + "+" if tok in (r"\d", r"\s", "[A-Za-z]") else tok * count for tok, count in runs
        )
        body = re.sub(r"\[A-Za-z\]\+(?:\\s\+\[A-Za-z\]\+)*", "[A-Za-z ]+", body)
    else:
        body = "".join(f"{tok}{{{count}}}" if count > 1 else tok for tok, count in runs)
    return "^" + body + "$"


def _pct(part: float, whole: float) -> str:
    return f"{(part / whole) * 100:.1f}" if whole else "0.0"


def _num(value: float) -> str:
    return f"{value:.4g}"


def _value_list(counts: pd.Series) -> List[Dict[str, str]]:
    return [{"value": str(k), "count": str(int(v))} for k, v in counts.items()]


def _shapes(text: pd.Series) -> Dict[str, Any]:
    """The top shapes, and the values outside the main one when it covers 95% or more.

    The exact shape is tried first; when it is not dominant the loose one is. Values outside the
    dominant shape are listed rarest first: a value seen once is more likely a typo than one seen
    twenty times.
    """
    sample = text.head(SHAPE_SAMPLE)
    total = len(sample)
    first: Dict[str, Any] = {}
    for loose in (False, True):
        shapes = sample.map(lambda v: value_shape(v, loose))
        counts = shapes.value_counts()
        result: Dict[str, Any] = {
            "shape_count": str(len(counts)),
            "shapes": [
                {"pattern": str(p), "pct": _pct(c, total)} for p, c in counts.head(TOP_SHAPES).items()
            ],
            "odd_count": 0,
            "odd": [],
            "dominant": "",
            "dominant_pct": "",
        }
        if not first:
            first = result
        top_pct = counts.iloc[0] / total * 100 if total else 0.0
        if top_pct >= DOMINANT_SHAPE_PCT:
            odd = sample[shapes != counts.index[0]]
            result["dominant"] = str(counts.index[0])
            result["dominant_pct"] = f"{top_pct:.1f}"
            result["odd_count"] = int(len(odd))
            result["odd"] = _value_list(odd.value_counts().sort_values(kind="stable").head(SHOWN_VALUES))
            return result
    return first


def _histogram(numbers: pd.Series) -> List[Dict[str, str]]:
    try:
        counts = pd.cut(numbers, bins=HIST_BINS, include_lowest=True, duplicates="drop").value_counts()
    except (ValueError, TypeError):
        return []
    counts = counts.sort_index()
    top = max(1, int(counts.max())) if len(counts) else 1
    bars = []
    for interval, count in counts.items():
        height = int(count / top * 100)
        bars.append(
            {
                "bin": f"{_num(interval.left)} - {_num(interval.right)}: {int(count)}",
                "height_pct": str(max(4, height) if count else 0),
            }
        )
    return bars


def _finding(title: str, count: int, rule: str, values: List[Dict[str, str]]) -> Dict[str, Any]:
    return {"title": f"{title} · {count:,}", "rule": rule, "values": values}


def column_profile(series: pd.Series, semantic_type: str, n_rows: int) -> Dict[str, Any]:
    """Everything the Profile view shows for one column, apart from its color and labels."""
    non_null = series.dropna()
    text = non_null.astype(str)
    filled = len(non_null)
    missing = int(series.isna().sum())
    unique = int(non_null.nunique(dropna=True))
    repeated = filled - unique
    lengths = text.str.len()
    semantic_type = str(semantic_type or "UNKNOWN")

    row: Dict[str, Any] = {
        "semantic_type": semantic_type,
        "missing_count": str(missing),
        "missing_pct": _pct(missing, max(1, n_rows)),
        "unique_count": str(unique),
        "unique_pct": _pct(unique, max(1, filled)),
        "repeated_count": str(repeated),
        "repeated_pct": _pct(repeated, max(1, filled)),
        "length_range": "",
        "length_median": "",
        "top_values": [
            {"value": str(k), "count": str(int(v)), "pct": _pct(int(v), filled)}
            for k, v in text.value_counts().head(TOP_VALUES).items()
        ],
        "is_numeric": "0",
        "numeric_min": "",
        "numeric_max": "",
        "numeric_mean": "",
        "numeric_median": "",
        "histogram": [],
    }
    if filled:
        low, high = int(lengths.min()), int(lengths.max())
        row["length_range"] = str(low) if low == high else f"{low}–{high}"
        row["length_median"] = f"med {int(lengths.median())}"

    findings: List[Dict[str, Any]] = []
    shapes = _shapes(text) if filled else {"shape_count": "0", "shapes": [], "odd_count": 0, "odd": []}
    row["shape_count"] = shapes["shape_count"]
    row["shapes"] = shapes["shapes"]
    if shapes["odd_count"]:
        findings.append(
            _finding(
                "Odd values",
                shapes["odd_count"],
                f"not {shapes['dominant']} ({shapes['dominant_pct']}%)",
                shapes["odd"],
            )
        )

    is_number_column = pd.api.types.is_numeric_dtype(series) or semantic_type.upper() in NUMERIC_TYPES
    if is_number_column and filled:
        strict = pd.to_numeric(text, errors="coerce")
        not_number = int(strict.isna().sum())
        if not_number and not_number < filled:
            findings.append(
                _finding(
                    "Not a number",
                    not_number,
                    f"{_pct(filled - not_number, filled)}% read as a number",
                    _value_list(text[strict.isna()].value_counts().head(SHOWN_VALUES)),
                )
            )
        # A dirty number such as "0.05%" or "12.0 oz" still counts with the number inside it.
        numbers = pd.to_numeric(text.str.extract(_NUMBER_IN_TEXT)[0], errors="coerce").dropna()
        if not numbers.empty:
            row["is_numeric"] = "1"
            row["numeric_min"] = _num(numbers.min())
            row["numeric_max"] = _num(numbers.max())
            row["numeric_mean"] = _num(numbers.mean())
            row["numeric_median"] = _num(numbers.median())
            row["histogram"] = _histogram(numbers)
            q1, q3 = numbers.quantile(0.25), numbers.quantile(0.75)
            spread = q3 - q1
            low_fence, high_fence = q1 - 1.5 * spread, q3 + 1.5 * spread
            far = numbers[(numbers < low_fence) | (numbers > high_fence)]
            # An id is a label, not a quantity: a far-off id is not a finding.
            if len(far) and semantic_type.upper() != "IDENTIFIER":
                far_counts = far.map(_num).value_counts().sort_values(kind="stable")
                findings.append(
                    _finding(
                        "Far from the rest",
                        int(len(far)),
                        f"outside {_num(low_fence)} – {_num(high_fence)}",
                        _value_list(far_counts.head(SHOWN_VALUES)),
                    )
                )

    placeholders = text[text.str.strip().str.lower().isin(PLACEHOLDERS)]
    if len(placeholders):
        findings.append(
            _finding("Placeholder text", int(len(placeholders)), "", _value_list(placeholders.value_counts()))
        )

    spaced = text[(text != text.str.strip()) | text.str.contains("  ", regex=False)]
    if len(spaced):
        examples = spaced.value_counts().head(SHOWN_VALUES)
        findings.append(
            _finding(
                "Extra spaces",
                int(len(spaced)),
                "leading, trailing or double",
                [{"value": repr(str(k)), "count": str(int(v))} for k, v in examples.items()],
            )
        )

    if unique / max(1, filled) >= 0.95 and repeated > 0:
        repeats = text.value_counts()
        repeats = repeats[repeats > 1]
        findings.append(
            _finding(
                "Repeats in a mostly unique column",
                repeated,
                f"{row['unique_pct']}% unique",
                _value_list(repeats.head(SHOWN_VALUES)),
            )
        )

    row["findings"] = findings
    row["finding_count"] = str(len(findings))
    row["has_findings"] = "1" if findings else "0"
    return row


def profile_rows(
    df: pd.DataFrame,
    columns: List[str],
    semantic_types: Dict[str, str],
) -> List[Dict[str, Any]]:
    """One profile per column, in the given order; columns missing from the table are skipped."""
    n_rows = len(df)
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        row = column_profile(df[col], semantic_types.get(col, "UNKNOWN"), n_rows)
        row["column"] = col
        rows.append(row)
    return rows
