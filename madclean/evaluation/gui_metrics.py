"""The evaluation numbers the GUI shows, computed by the one scorer.

The GUI scores whatever a person uploaded: no declared schema, so every column is tried as a
number, which is what its own scorer did cell by cell. The names below are the GUI's and the
dictionary is the one its page already renders; only the arithmetic underneath changed, from a
Python loop over every cell to the masks the scorer builds.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from madclean.evaluation.comparison import Mode, equal_mask
from madclean.evaluation.scoring import count_scores


def check_frames_compatible(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    *,
    other_label: str = "dataset",
) -> tuple[bool, str]:
    """Whether two frames can be compared at all, and a message saying why not."""
    if len(other) != len(reference):
        return False, (
            f"The {other_label} has {len(other)} rows, the reference has {len(reference)}. "
            "They must have the same number of rows."
        )
    if list(other.columns) != list(reference.columns):
        return False, (
            f"The {other_label} has different columns from the reference. "
            "They must have the same column names in the same order."
        )
    return True, ""


def compute_cleaning_metrics(
    dirty: pd.DataFrame,
    cleaned: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, Any]:
    """Cell-level metrics against ground truth, in the shape the GUI reads.

    - cell_accuracy: the share of cells where the cleaned value matches the ground truth.
    - repair_precision: of the cells the run changed, the share that now match.
    - repair_recall: of the cells that needed a repair, the share that now match.
    - f1_repair: the harmonic mean of the two, when both are defined.
    """
    # The three masks each cell is judged by, built once and shared by the table and its columns.
    errors = ~equal_mask(dirty, ground_truth, None, Mode.PAPER)
    changes = ~equal_mask(dirty, cleaned, None, Mode.PAPER)
    correct = equal_mask(cleaned, ground_truth, None, Mode.PAPER)

    rows = len(dirty)
    columns = list(dirty.columns)
    cells = rows * len(columns)

    per_column = {
        column: _view(
            count_scores(errors[column], changes[column], correct[column], rows),
            rows,
            int(correct[column].sum()),
        )
        for column in columns
    }
    metrics = _view(
        count_scores(errors, changes, correct, cells), cells, int(correct.to_numpy().sum())
    )
    metrics.update(
        {
            "n_rows": rows,
            "n_columns": len(columns),
            "n_cells": cells,
            "cells_correct": int(correct.to_numpy().sum()),
            "per_column": per_column,
        }
    )
    # The GUI calls the overall repair F1 by a different name from the per-column one.
    metrics["f1_repair"] = metrics.pop("f1")
    return metrics


def _view(scores, total_cells: int, correct_cells: int) -> dict[str, Any]:
    """One `Scores` in the GUI's names, with undefined scores as None rather than 0.0."""
    detection, correction = scores.detection_counts, scores.correction_counts
    changed = detection.total_changes
    needed = detection.total_errors
    precision = _ratio(correction.correctly_repaired_cells, changed)
    recall = _ratio(correction.correctly_repaired_cells, needed)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "tp": detection.true_positives,
        "fp": detection.false_positives,
        "tn": detection.true_negatives,
        "fn": detection.false_negatives,
        "cell_accuracy": _ratio(correct_cells, total_cells),
        "repair_precision": precision,
        "repair_recall": recall,
        "f1": f1,
        "cells_need_repair": needed,
        "cells_changed": changed,
    }


def _ratio(part: int, whole: int) -> float | None:
    """A share, or None when there is nothing to take a share of."""
    if whole == 0:
        return None
    return float(part) / float(whole)


def format_pct(x: float | None) -> str:
    """A share as a percentage for display, or a dash when it is undefined."""
    if x is None:
        return "—"
    return f"{100.0 * x:.2f}%"


def format_int(x: int) -> str:
    """A count for display."""
    return str(int(x))
