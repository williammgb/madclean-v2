"""Checks around the Coder's code, and the code the system writes itself from the Recommender's tables.

The Recommender's examples become test cases the code must reproduce; a guard rejects code that empties
cells that held real values; a value table is written into the final code as a `_VALUE_MAP`; and a
dependency's correction table is turned into code without a Coder.
"""
import re

import pandas as pd

from madclean.utils.helpers import align_dirty_cleaned_series

ARROWS = (" → ", " -> ")
# Words that stand for "no value" in the Recommender's examples.
MISSING_WORDS = {"nan", "nat", "none", "null", "empty", "<empty>", ""}
# Cells holding these (trimmed, lower-cased) may become empty without counting against the code.
PLACEHOLDER_WORDS = {
    "n/a", "na", "nan", "none", "null", "empty", "blank", "-", "--", "?", "missing", "unknown", "not available",
    "tbd", "#n/a", "",
}
MAX_VALUE_MAP_ENTRIES = 500
VALUE_MAP_ENTRIES_IN_PROMPT = 50
NUMBER_TOLERANCE = 1e-9


def is_empty(value) -> bool:
    """Missing, or text that is only spaces."""
    if value is None:
        return True
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


def shown(value) -> str:
    return "<empty>" if is_empty(value) else str(value)


def quoted(value) -> str:
    """A table value as the prompts and the trace print it: in quotes, or <empty> for None."""
    return "<empty>" if value is None else f'"{value}"'


def _unquote(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _is_missing_word(text: str) -> bool:
    return text.strip().lower() in MISSING_WORDS


# ---------------------------------------------------------------------------------------------
# Value table
# ---------------------------------------------------------------------------------------------


def value_table(value_mapping) -> tuple[dict[str, str | None], int]:
    """The Recommender's value changes as {from_value: to_value}, the first MAX_VALUE_MAP_ENTRIES of them,
    and how many entries were left out. A to_value of None means "make empty"."""
    table: dict[str, str | None] = {}
    entries = [entry for entry in (value_mapping or []) if isinstance(entry, dict) and entry.get("from_value") is not None]
    for entry in entries[:MAX_VALUE_MAP_ENTRIES]:
        to_value = entry.get("to_value")
        table[str(entry["from_value"])] = None if to_value is None else str(to_value)
    return table, max(0, len(entries) - MAX_VALUE_MAP_ENTRIES)


def value_table_for_prompt(value_mapping) -> str:
    """The table as the Coder sees it: up to VALUE_MAP_ENTRIES_IN_PROMPT lines, then a count."""
    table, _ = value_table(value_mapping)
    if not table:
        return "(none)"
    items = list(table.items())
    lines = [f"- {quoted(old)} → {quoted(new)}" for old, new in items[:VALUE_MAP_ENTRIES_IN_PROMPT]]
    rest = len(items) - VALUE_MAP_ENTRIES_IN_PROMPT
    if rest > 0:
        lines.append(f"… and {rest} more replacements")
    return "\n".join(lines)


_DEF_CLEAN_COLUMN = re.compile(r"\bdef\s+clean_column\s*\(")

_VALUE_MAP_WRAPPER = '''
def clean_column(column: pd.Series{base_parameter}) -> pd.Series:
    _VALUE_MAP = {value_map}
    cleaned = {cleaned}
    original = column.astype(str).where(column.notna())
    hit = original.isin(list(_VALUE_MAP)).to_numpy()
    if not hit.any() or len(cleaned) != len(column):
        return cleaned
    values = cleaned.astype(object).to_numpy(copy=True)
    values[hit] = [_VALUE_MAP[value] for value in original[hit]]
    result = pd.Series(values, index=cleaned.index, name=cleaned.name)
    result = result.where(result.notna())
    if pd.api.types.is_numeric_dtype(cleaned) and not pd.api.types.is_bool_dtype(cleaned):
        numbers = pd.to_numeric(result, errors="coerce")
        if numbers.notna().sum() == result.notna().sum():
            return numbers
    return result
'''


def wrap_with_value_map(code: str | None, value_mapping) -> str:
    """The final code: the Coder's function (or none), then the value table applied on the original values.

    The Coder's `clean_column` is renamed `_clean_by_code` and handed to the wrapping `clean_column` as a
    default argument, so each column's notebook cell keeps its own function and its own `_VALUE_MAP`.
    """
    table, _ = value_table(value_mapping)
    if not table:
        return code or ""
    value_map = "{\n" + "".join(f"        {old!r}: {new!r},\n" for old, new in table.items()) + "    }"
    if code:
        base = _DEF_CLEAN_COLUMN.sub("def _clean_by_code(", code.strip(), count=1)
        head = base if "import pandas as pd" in base else "import pandas as pd\n\n" + base
        wrapper = _VALUE_MAP_WRAPPER.format(
            base_parameter=", _clean_by_code=_clean_by_code",
            value_map=value_map,
            cleaned="_clean_by_code(column.copy())",
        )
        return head + "\n\n" + wrapper.strip() + "\n"
    wrapper = _VALUE_MAP_WRAPPER.format(base_parameter="", value_map=value_map, cleaned="column.copy()")
    return "import pandas as pd\n\n\n" + wrapper.strip() + "\n"


def keep_whole_numbers(original: pd.Series, cleaned: pd.Series) -> pd.Series:
    """`cleaned` as whole numbers when its only decimals are whole numbers written as 64.0.

    That happens when loading reads a whole-number column with blank cells as decimals, or when the
    cleaning code returns decimals. A decimal column whose filled values are all whole goes back to whole
    numbers when the original numbers were whole too, or were mostly written without a decimal point.
    Missing cells stay missing.
    """
    if not pd.api.types.is_float_dtype(cleaned):
        return cleaned
    filled = cleaned.dropna()
    if filled.empty or not (filled == filled.round()).all() or filled.abs().max() >= 2**53:
        return cleaned
    before = original.dropna()
    if pd.api.types.is_bool_dtype(before) or before.empty:
        return cleaned
    if pd.api.types.is_numeric_dtype(before):
        # Read as decimals only because some cells are blank: 64 is loaded as 64.0.
        whole_before = bool((before == before.round()).all())
    else:
        text = before.astype(str).str.strip()
        numbers = text[pd.to_numeric(text, errors="coerce").notna()]
        whole_before = not numbers.empty and numbers.str.contains(r"[.eE]").mean() < 0.5
    return cleaned.astype("Int64") if whole_before else cleaned


# ---------------------------------------------------------------------------------------------
# Example check
# ---------------------------------------------------------------------------------------------


def parse_example(item) -> tuple[str, str | None] | None:
    """(input, expected) from "dirty → clean", expected None meaning "expect empty".

    Split on the first arrow; None when there is no arrow or the input is itself a missing value.
    """
    text = str(item)
    found = [(text.find(arrow), arrow) for arrow in ARROWS if arrow in text]
    if not found:
        return None
    position, arrow = min(found)
    source, target = _unquote(text[:position]), _unquote(text[position + len(arrow):])
    if _is_missing_word(source):
        return None
    return source, (None if _is_missing_word(target) else target)


def example_cases(recommender_data: dict | None) -> list[tuple[str, str | None]]:
    """The cases the code must reproduce: each dirty example, and each clean example unchanged.

    Inputs the value table covers are left out: the table is applied last and wins.
    """
    data = recommender_data or {}
    table, _ = value_table(data.get("value_mapping"))
    cases: list[tuple[str, str | None]] = []
    for item in data.get("examples_dirty") or []:
        case = parse_example(item)
        if case is not None:
            cases.append(case)
    for item in data.get("examples_clean") or []:
        value = _unquote(str(item))
        if not _is_missing_word(value):
            cases.append((value, value))
    return [case for case in cases if case[0] not in table]


def emptied_inputs(recommender_data: dict | None) -> list[str]:
    """Inputs the Recommender itself says become empty, through its examples or its value table."""
    data = recommender_data or {}
    inputs = [case[0] for case in map(parse_example, data.get("examples_dirty") or []) if case and case[1] is None]
    table, _ = value_table(data.get("value_mapping"))
    return inputs + [old for old, new in table.items() if new is None]


def example_inputs(column: pd.Series, inputs: list[str]) -> pd.Series:
    """The inputs in the column's own dtype when every one converts to it, otherwise as text."""
    dtype = column.dtype
    if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
        try:
            numbers = pd.to_numeric(pd.Series(inputs, dtype=object), errors="raise")
            if pd.api.types.is_integer_dtype(dtype) and not all(float(n).is_integer() for n in numbers):
                raise ValueError("not whole numbers")
            return numbers.astype(dtype)
        except (ValueError, TypeError):
            pass
    return pd.Series(inputs, dtype=object)


def _as_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def values_match(got, expected: str | None) -> bool:
    """Both empty; or both numbers within NUMBER_TOLERANCE; or equal text after trimming spaces."""
    if expected is None or is_empty(got):
        return expected is None and is_empty(got)
    got_number, expected_number = _as_number(got), _as_number(expected)
    if got_number is not None and expected_number is not None:
        return abs(got_number - expected_number) <= NUMBER_TOLERANCE
    return str(got).strip() == str(expected).strip()


def example_failures(result, cases: list[tuple[str, str | None]]) -> list[tuple[str, str | None, str]]:
    """(input, expected, what the code gave) for every case the result gets wrong.

    `result` is what the code returned for the inputs in order, or an error message when it did not run.
    """
    if not isinstance(result, pd.Series) or len(result) != len(cases):
        gave = f"error: {result}" if isinstance(result, str) else "no value per input"
        return [(source, expected, gave) for source, expected in cases]
    return [
        (source, expected, shown(got))
        for (source, expected), got in zip(cases, result.tolist())
        if not values_match(got, expected)
    ]


def example_feedback(failures: list[tuple[str, str | None, str]], total: int) -> str:
    lines = [f"- {source} → {shown(expected)} → {gave}" for source, expected, gave in failures]
    return (
        f"Example check: your code gave the wrong value for {len(failures)} of the Recommender's {total} examples.\n\n"
        "Cases (input → expected → your code gave):\n" + "\n".join(lines) + "\n\n"
        "Fix the code so that every example gives its expected value, and provide the full, fixed executable "
        "code string again."
    )


def example_note(failed: int, total: int) -> str:
    return f"Example check: {total - failed} of {total} passed"


# ---------------------------------------------------------------------------------------------
# Emptied-cells guard
# ---------------------------------------------------------------------------------------------


def emptied_share(original: pd.Series, cleaned: pd.Series, empty_inputs=()) -> tuple[int, int]:
    """(cells emptied, cells filled before), leaving out placeholders and inputs meant to become empty.

    Only cells that become empty count; how many cells change is never looked at.
    """
    original, cleaned = align_dirty_cleaned_series(original, cleaned)
    allowed = PLACEHOLDER_WORDS | {str(value).strip().lower() for value in empty_inputs}
    filled = emptied = 0
    for before, after in zip(original.tolist(), cleaned.tolist()):
        if is_empty(before) or str(before).strip().lower() in allowed:
            continue
        filled += 1
        emptied += is_empty(after)
    return emptied, filled


def emptied_feedback(emptied: int, filled: int) -> str:
    share = 100.0 * emptied / filled if filled else 0.0
    return (
        f"Your code emptied {emptied} of {filled} filled cells ({share:.1f}%). "
        "Only missing-value placeholders may become empty."
    )


# ---------------------------------------------------------------------------------------------
# Dependency table
# ---------------------------------------------------------------------------------------------


def _rhs_value(value, rhs_dtype):
    """correct_rhs as the right-hand column holds it: a number when that column is numeric, else None if
    it is not one; the text otherwise."""
    if not pd.api.types.is_numeric_dtype(rhs_dtype) or pd.api.types.is_bool_dtype(rhs_dtype):
        return str(value)
    number = _as_number(value)
    if number is None:
        return None
    return int(number) if pd.api.types.is_integer_dtype(rhs_dtype) and number.is_integer() else number


def dependency_table(corrections, rhs_dtype) -> tuple[dict, list[str]]:
    """{lhs_value: correct_rhs} from the Recommender's corrections, and the lhs values whose correct_rhs
    is not a number for a numeric right-hand column."""
    table: dict = {}
    unusable: list[str] = []
    for entry in corrections or []:
        if not isinstance(entry, dict) or entry.get("lhs_value") is None or entry.get("correct_rhs") is None:
            continue
        value = _rhs_value(entry["correct_rhs"], rhs_dtype)
        if value is None:
            unusable.append(str(entry["lhs_value"]))
            continue
        table[str(entry["lhs_value"])] = value
    return table, unusable


def unmatched_lhs_values(lhs_column: pd.Series, table: dict) -> list[str]:
    """Table keys that match no row's left-hand value written as text."""
    present = set(lhs_column.astype(str).where(lhs_column.notna()).dropna())
    return [key for key in table if key not in present]


def dependency_code_from_table(lhs: str, rhs: str, table: dict, impute_missing: bool) -> str:
    """Code that sets rhs to the table's value on filled rows whose lhs (as text) is a key, then, if asked,
    fills empty rhs cells whose lhs has exactly one known rhs value. Only imputation fills empty cells."""
    lines = [
        "import pandas as pd",
        "",
        "",
        "def clean_column(df: pd.DataFrame) -> pd.DataFrame:",
        "    df = df.copy()",
    ]
    if table:
        corrections = "{\n" + "".join(f"        {key!r}: {value!r},\n" for key, value in table.items()) + "    }"
        lines += [
            f"    corrections = {corrections}",
            f"    lhs_text = df[{lhs!r}].astype(str).where(df[{lhs!r}].notna())",
            f"    fix = lhs_text.isin(list(corrections)) & df[{rhs!r}].notna()",
            f"    df.loc[fix, {rhs!r}] = lhs_text[fix].map(corrections)",
        ]
    if impute_missing:
        lines += [
            f"    known = df.dropna(subset=[{lhs!r}, {rhs!r}]).groupby({lhs!r})[{rhs!r}].unique()",
            "    single = {key: values[0] for key, values in known.items() if len(values) == 1}",
            f"    fill = df[{rhs!r}].isna() & df[{lhs!r}].isin(list(single))",
            f"    df.loc[fill, {rhs!r}] = df.loc[fill, {lhs!r}].map(single)",
        ]
    lines.append("    return df")
    return "\n".join(lines) + "\n"
