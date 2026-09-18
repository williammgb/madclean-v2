"""Scoring one cleaning run against its ground truth.

Three frames go in — the dirty input, the cleaned output and the ground truth — and the scores
come out of two comparisons between them: which cells were wrong to begin with, and which cells
the cleaning changed. Detection asks whether the cleaning touched the right cells; correction asks
whether what it wrote is what the ground truth holds.

The counting is the thesis's, so paper mode still reproduces the committed results. What is new is
that the comparison it counts is chosen (see `comparison.Mode`), and that the same class now serves
the command line, the benchmark scripts and the GUI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from madclean.evaluation.comparison import Mode, equal_mask
from madclean.evaluation.scores import (
    CorrectionCounts,
    DetectionCounts,
    PrecisionRecallF1,
    RunEvaluation,
    Scores,
)
from madclean.utils.helpers import load_dataset

# A detailed report lists this many rows per column at most, as in the thesis.
MAX_REPORT_ROWS = 1500


class Evaluator:
    """Scores cleaned frames against one dirty/ground-truth pair.

    `numeric_columns` names the columns a benchmark declares numeric. None — the GUI's case, where
    an uploaded file comes with no schema — compares every column as numbers where it can.
    """

    def __init__(
        self,
        dirty: pd.DataFrame | str | Path,
        ground_truth: pd.DataFrame | str | Path,
        numeric_columns: set[str] | None = None,
        mode: Mode | str = Mode.PAPER,
    ):
        self.dirty = _as_frame(dirty)
        self.ground_truth = _as_frame(ground_truth)
        self.numeric_columns = numeric_columns
        self.mode = Mode(mode)
        self._check_shapes()
        # Where the dirty input disagrees with the ground truth: the errors the run had to fix.
        self.errors = ~self._equal(self.dirty, self.ground_truth)

    def evaluate(self, cleaned: pd.DataFrame | str | Path) -> RunEvaluation:
        """Scores one cleaned frame: the table as a whole, each column, and what changed."""
        cleaned = _as_frame(cleaned)
        if cleaned.shape != self.dirty.shape:
            raise ValueError("The cleaned frame must have the same shape as the dirty frame.")
        changes = ~self._equal(self.dirty, cleaned)
        correct = self._equal(cleaned, self.ground_truth)
        overall = count_scores(self.errors, changes, correct, self.dirty.size)
        per_column = {
            column: count_scores(
                self.errors[column], changes[column], correct[column], len(self.dirty[column])
            )
            for column in self.dirty.columns
        }
        return RunEvaluation(
            overall=overall,
            per_column=per_column,
            mode=self.mode.value,
            reports=self._reports(cleaned, changes, correct),
        )

    def _equal(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        return equal_mask(left, right, self.numeric_columns, self.mode)

    def _check_shapes(self) -> None:
        if self.dirty.shape != self.ground_truth.shape:
            raise ValueError("The dirty and ground truth frames must have the same shape.")
        if not self.dirty.index.equals(self.ground_truth.index) or not self.dirty.columns.equals(
            self.ground_truth.columns
        ):
            raise ValueError("The dirty and ground truth frames must have the same rows and columns.")

    def _reports(
        self, cleaned: pd.DataFrame, changes: pd.DataFrame, correct: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """One frame per column listing every cell that was wrong or was touched, and how it ended."""
        reports: dict[str, pd.DataFrame] = {}
        for column in self.dirty.columns:
            interesting = self.errors[column] | changes[column]
            if not interesting.any():
                continue
            report = pd.DataFrame(
                {
                    "dirty": self.dirty.loc[interesting, column],
                    "cleaned": cleaned.loc[interesting, column],
                    "ground_truth": self.ground_truth.loc[interesting, column],
                }
            )
            report["status"] = [
                _status(
                    dirty_was_right=not was_error,
                    cleaned_is_right=is_correct,
                    value_untouched=not was_changed,
                )
                for was_error, is_correct, was_changed in zip(
                    self.errors.loc[interesting, column],
                    correct.loc[interesting, column],
                    changes.loc[interesting, column],
                )
            ]
            reports[column] = report.head(MAX_REPORT_ROWS)
        return reports


def _status(*, dirty_was_right: bool, cleaned_is_right: bool, value_untouched: bool) -> str:
    """The thesis's wording for how one cell turned out."""
    if not dirty_was_right and cleaned_is_right:
        return "Corrected (TP)"
    if dirty_was_right and not cleaned_is_right:
        return "Wrong correction (FP)"
    if not dirty_was_right and not cleaned_is_right and not value_untouched:
        return "Detected but wrong fix"
    if not dirty_was_right and value_untouched:
        return "Not fixed (FN)"
    return "(TN)"


def count_scores(
    errors: pd.DataFrame | pd.Series,
    changes: pd.DataFrame | pd.Series,
    correct: pd.DataFrame | pd.Series,
    total_cells: int,
) -> Scores:
    """Turns three masks into the counts and the scores.

    Detection: a change to a cell that was wrong is a hit, a change to a cell that was right is a
    false alarm, a cell left wrong is a miss. Correction: of the cells the run both detected and
    changed, how many now hold the ground truth's value.
    """
    errors = errors.to_numpy(dtype=bool)
    changes = changes.to_numpy(dtype=bool)
    correct = correct.to_numpy(dtype=bool)

    true_positives = int(np.sum(errors & changes))
    false_positives = int(np.sum(changes & ~errors))
    false_negatives = int(np.sum(errors & ~changes))
    true_negatives = int(total_cells - (true_positives + false_positives + false_negatives))

    repaired = int(np.sum(correct & errors & changes))
    total_changes = int(np.sum(changes))
    total_errors = int(np.sum(errors))

    return Scores(
        detection_counts=DetectionCounts(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            true_negatives=true_negatives,
            total_errors=total_errors,
            total_changes=total_changes,
        ),
        detection_metrics=precision_recall_f1(
            true_positives, true_positives + false_positives, true_positives + false_negatives
        ),
        correction_counts=CorrectionCounts(
            correctly_repaired_cells=repaired,
            incorrectly_repaired_cells=true_positives - repaired,
            repaired_clean_cells=false_positives,
        ),
        correction_metrics=precision_recall_f1(repaired, total_changes, total_errors),
    )


def precision_recall_f1(hits: int, claimed: int, wanted: int) -> PrecisionRecallF1:
    """Precision, recall and F1. Undefined counts as 0.0, as in the paper."""
    precision = hits / claimed if claimed > 0 else 0.0
    recall = hits / wanted if wanted > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return PrecisionRecallF1(precision=float(precision), recall=float(recall), f1_score=float(f1))


def _as_frame(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    return source if isinstance(source, pd.DataFrame) else load_dataset(source)
