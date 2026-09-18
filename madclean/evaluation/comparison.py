"""Deciding whether two cells hold the same value.

Every score in this package comes out of one question asked cell by cell: does this value match
that one? The answer depends on how strict the comparison is, and this module is the only place
that decides it.

`paper` is the comparison the thesis used, moved here unchanged, because the committed results are
the record this system is judged against: values that are missing on both sides match, a value
missing on one side does not, columns a dataset declares numeric compare as numbers within a
relative tolerance of 1e-9, and everything else compares with `==`.

`strict` asks for the type as well. A cleaning run that leaves "20" where the ground truth holds
the number 20 has not finished the job, and paper mode cannot see the difference. Strict mode
counts it as an error.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

# The tolerance the thesis compared numbers with; kept so paper mode reproduces its results.
NUMERIC_TOLERANCE = 1e-9

# Types that cannot be compared elementwise by numpy and are compared as text instead.
_COMPLEX_TYPES = (list, np.ndarray, dict, set)


class Mode(str, Enum):
    """How hard a comparison is to satisfy."""

    PAPER = "paper"
    STRICT = "strict"


def equal_mask(
    left: pd.DataFrame,
    right: pd.DataFrame,
    numeric_columns: set[str] | None = None,
    mode: Mode | str = Mode.PAPER,
) -> pd.DataFrame:
    """A True/False frame saying, cell by cell, whether `left` and `right` hold the same value.

    `numeric_columns` names the columns compared as numbers in paper mode. None means every column
    is tried as a number first, which is what an uploaded dataset with no declared schema gets.
    The result never contains NA: a comparison that cannot be made is False.
    """
    mode = Mode(mode)
    result = pd.DataFrame(False, index=left.index, columns=left.columns, dtype=bool)
    for column in left.columns:
        numeric = numeric_columns is None or column in numeric_columns
        result[column] = _equal_column(
            left[column].to_numpy(),
            right[column].to_numpy(),
            numeric=numeric,
            mode=mode,
            ground_truth=right[column],
        )
    return result


def _equal_column(
    left: np.ndarray,
    right: np.ndarray,
    *,
    numeric: bool,
    mode: Mode,
    ground_truth: pd.Series,
) -> np.ndarray:
    """Compares one column, handling missing values before anything else looks at the data."""
    left_missing = pd.isna(left)
    right_missing = pd.isna(right)
    both_missing = left_missing & right_missing
    both_present = ~left_missing & ~right_missing

    equal = np.zeros(len(left), dtype=bool)
    if both_present.any():
        present_left = left[both_present]
        present_right = right[both_present]
        if mode is Mode.STRICT:
            decided = _strict_equal(present_left, present_right, ground_truth)
        elif numeric:
            decided = _numeric_equal(present_left, present_right)
        else:
            decided = _plain_equal(present_left, present_right)
        equal[both_present] = decided

    # A value missing on one side only stays False: the cleaning either dropped it or invented it.
    return both_missing | equal


def _numeric_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compares as numbers where both sides parse as numbers, and as values where they do not."""
    equal = np.zeros(len(left), dtype=bool)
    try:
        left_numbers = pd.to_numeric(left, errors="coerce")
        right_numbers = pd.to_numeric(right, errors="coerce")
    except (TypeError, ValueError):
        return _plain_equal(left, right)

    both_numbers = ~pd.isna(left_numbers) & ~pd.isna(right_numbers)
    if both_numbers.any():
        equal[both_numbers] = np.isclose(
            left_numbers[both_numbers], right_numbers[both_numbers], rtol=NUMERIC_TOLERANCE
        )
    not_numbers = ~both_numbers
    if not_numbers.any():
        equal[not_numbers] = _plain_equal(left[not_numbers], right[not_numbers])
    return equal


def _plain_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compares values as they are, falling back to text for types numpy will not compare."""
    try:
        return np.asarray(left == right, dtype=bool)
    except ValueError:
        equal = np.zeros(len(left), dtype=bool)
        for index in range(len(left)):
            one, other = left[index], right[index]
            if isinstance(one, _COMPLEX_TYPES) or isinstance(other, _COMPLEX_TYPES):
                equal[index] = str(one) == str(other)
            else:
                equal[index] = bool(one == other)
        return equal


def _strict_equal(left: np.ndarray, right: np.ndarray, ground_truth: pd.Series) -> np.ndarray:
    """Matches only when the value and the kind of value both agree with the ground truth.

    The kind comes from the ground truth column: a column pandas read as integers wants integers,
    one it read as decimals wants decimals, and an object column wants whatever that cell holds.
    """
    wanted = _column_kind(ground_truth)
    equal = np.zeros(len(left), dtype=bool)
    for index in range(len(left)):
        one, other = left[index], right[index]
        kind = wanted or _value_kind(other)
        if _value_kind(one) != kind:
            continue
        if kind == "float":
            equal[index] = bool(np.isclose(float(one), float(other), rtol=NUMERIC_TOLERANCE))
        elif kind in ("int", "bool"):
            equal[index] = one == other
        elif kind == "text":
            equal[index] = str(one) == str(other)
        else:
            equal[index] = _plain_equal(np.array([one], dtype=object), np.array([other], dtype=object))[0]
    return equal


def _column_kind(column: pd.Series) -> str | None:
    """The kind a whole column asks for, or None when the column holds mixed Python objects."""
    if pd.api.types.is_bool_dtype(column):
        return "bool"
    if pd.api.types.is_integer_dtype(column):
        return "int"
    if pd.api.types.is_float_dtype(column):
        return "float"
    if pd.api.types.is_string_dtype(column) and column.dtype != object:
        return "text"
    return None


def _value_kind(value: object) -> str:
    """The kind of one value. Booleans are checked first: in Python a bool is also an int."""
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating)):
        return "float"
    if isinstance(value, str):
        return "text"
    return "other"
