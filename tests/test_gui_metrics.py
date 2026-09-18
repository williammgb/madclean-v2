"""The numbers the web interface shows.

The GUI had its own scorer, a cell-by-cell Python loop with its own metric names. It is gone: the
same numbers now come from the one scorer. Before it was deleted, both were run side by side on
seven frames — text and numbers mixed, floats, missing values, an already-correct frame and four
random ones — and agreed on every field. This test pins the first of those frames so the shape and
the arithmetic cannot drift afterwards.
"""

import pandas as pd
import pytest

from madclean.evaluation import check_frames_compatible, compute_cleaning_metrics, format_pct

DIRTY = pd.DataFrame({"a": ["1O", "2", "x", None, "5"], "b": ["Anna", "bram", None, "Dirk", "eva"]})
GROUND_TRUTH = pd.DataFrame({"a": ["10", "2", "3", None, "5"], "b": ["Anna", "Bram", "Cees", "Dirk", "Eva"]})
CLEANED = pd.DataFrame({"a": ["10", "2", "3", None, "5"], "b": ["Anna", "Bram", None, "Dirk", "Eva"]})


def test_the_metrics_the_gui_reads_are_unchanged():
    metrics = compute_cleaning_metrics(DIRTY, CLEANED, GROUND_TRUTH)

    assert metrics["n_rows"] == 5
    assert metrics["n_columns"] == 2
    assert metrics["n_cells"] == 10
    assert metrics["cells_correct"] == 9
    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (4, 0, 5, 1)
    assert metrics["cell_accuracy"] == pytest.approx(0.9)
    assert metrics["repair_precision"] == pytest.approx(1.0)
    assert metrics["repair_recall"] == pytest.approx(0.8)
    assert metrics["f1_repair"] == pytest.approx(0.888888888, abs=1e-6)

    column_b = metrics["per_column"]["b"]
    # "Cees" was dropped rather than corrected: a miss, and one cell fewer correct.
    assert (column_b["tp"], column_b["fn"]) == (2, 1)
    assert column_b["repair_recall"] == pytest.approx(2 / 3)
    assert column_b["f1"] == pytest.approx(0.8)


def test_a_column_with_nothing_to_repair_reports_no_score_rather_than_zero():
    """The GUI prints a dash for an undefined score, so the metric has to be None, not 0.0."""
    frame = pd.DataFrame({"a": ["x", "y", "z"]})
    metrics = compute_cleaning_metrics(frame, frame.copy(), frame.copy())

    assert metrics["cells_need_repair"] == 0
    assert metrics["repair_recall"] is None
    assert metrics["repair_precision"] is None
    assert metrics["f1_repair"] is None
    assert format_pct(metrics["f1_repair"]) == "—"
    assert format_pct(metrics["cell_accuracy"]) == "100.00%"


def test_frames_that_cannot_be_compared_are_refused_with_a_reason():
    ok, message = check_frames_compatible(DIRTY, DIRTY.head(2), other_label="ground truth")
    assert ok is False
    assert "rows" in message and "ground truth" in message

    renamed = DIRTY.rename(columns={"a": "other"})
    ok, message = check_frames_compatible(DIRTY, renamed, other_label="cleaned file")
    assert ok is False
    assert "columns" in message

    ok, message = check_frames_compatible(DIRTY, GROUND_TRUTH)
    assert ok is True
    assert message == ""
