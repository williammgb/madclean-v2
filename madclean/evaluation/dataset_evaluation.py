"""
Compare dirty input, model-cleaned output, and ground-truth datasets.

Row alignment: all frames must have the same row count and column names (same order)
as the reference (typically the originally uploaded dirty file).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _cell_equal(a: Any, b: Any) -> bool:
    """Loose equality aligned with GUI cell comparison (numeric tolerance, NA)."""
    a_na = pd.isna(a)
    b_na = pd.isna(b)
    if bool(a_na) and bool(b_na):
        return True
    if bool(a_na) or bool(b_na):
        return False
    try:
        a_num = pd.to_numeric(a, errors="coerce")
        b_num = pd.to_numeric(b, errors="coerce")
        if not pd.isna(a_num) and not pd.isna(b_num):
            try:
                return math.isclose(float(a_num), float(b_num), rel_tol=1e-9, abs_tol=0.0)
            except Exception:
                pass
    except Exception:
        pass
    try:
        return a == b
    except Exception:
        return str(a) == str(b)


def check_frames_compatible(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    *,
    other_label: str = "dataset",
) -> tuple[bool, str]:
    """Require identical columns (names and order) and identical row counts."""
    if list(reference.columns) != list(other.columns):
        ref_cols = list(reference.columns)
        oth_cols = list(other.columns)
        return False, (
            f"{other_label} columns do not match the loaded dataset. "
            f"Expected {len(ref_cols)} columns in order {ref_cols[:12]}"
            f"{'...' if len(ref_cols) > 12 else ''}, "
            f"got {len(oth_cols)} columns {oth_cols[:12]}{'...' if len(oth_cols) > 12 else ''}."
        )
    if len(reference) != len(other):
        return False, (
            f"{other_label} row count ({len(other)}) does not match the loaded dataset ({len(reference)} rows)."
        )
    return True, ""


def compute_cleaning_metrics(
    dirty: pd.DataFrame,
    cleaned: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, Any]:
    """
    Cell-level metrics vs ground truth.

    - cell_accuracy: fraction of cells where cleaned matches GT.
    - repair_precision: among cells where cleaner changed the value (dirty != cleaned),
      fraction where cleaned matches GT.
    - repair_recall: among cells where GT differs from dirty (repair needed),
      fraction where cleaned matches GT.
    - f1_repair: harmonic mean of repair_precision and repair_recall when both defined.

    Per-column dicts use the same definitions on that column only.
    """
    cols = list(dirty.columns)
    n_rows = len(dirty)
    n_cells = n_rows * len(cols) if cols else 0

    correct = 0
    changed = 0
    changed_and_correct = 0
    need_fix = 0
    fixed_correct = 0

    # Detection confusion-matrix style counts (did we change vs should we change).
    tp = 0  # changed & needed fix
    fp = 0  # changed but no fix needed
    fn = 0  # no change but fix needed
    tn = 0  # no change and no fix needed

    per_col: dict[str, dict[str, float | int | None]] = {}

    for col in cols:
        c_correct = 0
        c_changed = 0
        c_changed_ok = 0
        c_need = 0
        c_fixed = 0
        c_tp = 0
        c_fp = 0
        c_fn = 0
        c_tn = 0
        ds = dirty[col]
        cs = cleaned[col]
        gs = ground_truth[col]
        for i in range(n_rows):
            d, cl, g = ds.iloc[i], cs.iloc[i], gs.iloc[i]
            is_equal_clean_gt = _cell_equal(cl, g)
            is_changed = not _cell_equal(d, cl)
            is_error = not _cell_equal(d, g)

            if is_equal_clean_gt:
                correct += 1
                c_correct += 1
            if is_changed:
                changed += 1
                c_changed += 1
                if is_equal_clean_gt:
                    changed_and_correct += 1
                    c_changed_ok += 1
            if is_error:
                need_fix += 1
                c_need += 1
                if is_equal_clean_gt:
                    fixed_correct += 1
                    c_fixed += 1

            # Detection confusion matrix (change vs should change).
            if is_error and is_changed:
                tp += 1
                c_tp += 1
            elif (not is_error) and is_changed:
                fp += 1
                c_fp += 1
            elif is_error and (not is_changed):
                fn += 1
                c_fn += 1
            else:
                tn += 1
                c_tn += 1

        def _safe_div(num: int, den: int) -> float | None:
            if den == 0:
                return None
            return float(num) / float(den)

        p = _safe_div(c_changed_ok, c_changed)
        r = _safe_div(c_fixed, c_need)
        f1 = None
        if p is not None and r is not None and (p + r) > 0:
            f1 = 2.0 * p * r / (p + r)
        per_col[col] = {
            "tp": c_tp,
            "fp": c_fp,
            "tn": c_tn,
            "fn": c_fn,
            "cell_accuracy": _safe_div(c_correct, n_rows) if n_rows else None,
            "repair_precision": p,
            "repair_recall": r,
            "f1": f1,
            "cells_need_repair": c_need,
            "cells_changed": c_changed,
        }

    cell_accuracy = _safe_div(correct, n_cells) if n_cells else None
    repair_precision = _safe_div(changed_and_correct, changed) if changed else None
    repair_recall = _safe_div(fixed_correct, need_fix) if need_fix else None
    f1_repair = None
    if (
        repair_precision is not None
        and repair_recall is not None
        and (repair_precision + repair_recall) > 0
    ):
        f1_repair = (
            2.0 * repair_precision * repair_recall / (repair_precision + repair_recall)
        )

    return {
        "n_rows": n_rows,
        "n_columns": len(cols),
        "n_cells": n_cells,
        "cells_correct": correct,
        "cells_changed": changed,
        "cells_need_repair": need_fix,
        # Detection confusion matrix (overall).
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "cell_accuracy": cell_accuracy,
        "repair_precision": repair_precision,
        "repair_recall": repair_recall,
        "f1_repair": f1_repair,
        "per_column": per_col,
    }


def format_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100.0 * x:.2f}%"


def format_int(x: int) -> str:
    return str(int(x))
