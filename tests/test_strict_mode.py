"""Strict mode: the type has to match too.

Paper mode was written to reproduce the thesis's numbers, and it compares "20" and 20 as the same
value. Strict mode does not. These tests pin the difference, and the property that matters when
both are reported side by side: strict mode can never flatter a run that paper mode judged.
"""

from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from madclean.evaluation import BENCHMARKS, Evaluator, Mode, equal_mask
from madclean.utils.helpers import load_dataset

CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_DIR / "evaluation" / "results"


def _frame(values):
    return pd.DataFrame({"column": values})


@pytest.mark.parametrize(
    ("cleaned", "ground_truth", "paper_matches", "strict_matches"),
    [
        ([20], [20], True, True),                    # same number, same type
        (["20"], [20], True, False),                 # the text of a number is not the number
        ([20.0], [20], True, False),                 # a decimal in an integer column
        (["Utrecht"], ["Utrecht"], True, True),      # text compares the same either way
        (["utrecht"], ["Utrecht"], False, False),    # neither mode ignores case
        ([None], [None], True, True),                # missing on both sides is a match
        ([None], [20], False, False),                # missing on one side never matches
    ],
)
def test_the_two_modes_differ_only_on_the_type(cleaned, ground_truth, paper_matches, strict_matches):
    left, right = _frame(cleaned), _frame(ground_truth)
    assert bool(equal_mask(left, right, None, Mode.PAPER).iloc[0, 0]) is paper_matches
    assert bool(equal_mask(left, right, None, Mode.STRICT).iloc[0, 0]) is strict_matches


def test_a_column_left_as_text_scores_worse_in_strict_mode():
    """The case strict mode exists for: values that are right, in the wrong form."""
    dirty = _frame(["1O", "2O", "3O"])          # the letter O instead of a zero
    ground_truth = _frame([10, 20, 30])
    cleaned = _frame(["10", "20", "30"])        # cleaned, but still text

    paper = Evaluator(dirty, ground_truth, None, Mode.PAPER).evaluate(cleaned)
    strict = Evaluator(dirty, ground_truth, None, Mode.STRICT).evaluate(cleaned)

    assert paper.overall.correction_metrics.f1_score == 1.0
    assert strict.overall.correction_metrics.f1_score == 0.0
    assert strict.overall.correction_counts.correctly_repaired_cells == 0


@settings(max_examples=40, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.one_of(st.integers(-50, 50), st.text(min_size=1, max_size=4), st.none()),
            st.one_of(st.integers(-50, 50), st.text(min_size=1, max_size=4), st.none()),
            st.one_of(st.integers(-50, 50), st.text(min_size=1, max_size=4), st.none()),
        ),
        min_size=1,
        max_size=12,
    )
)
def test_strict_mode_never_calls_more_cells_equal_than_paper_mode(rows):
    """Whatever the data, strict mode is the harder question: every match it finds, paper finds too."""
    left = _frame([row[0] for row in rows])
    right = _frame([row[1] for row in rows])

    paper = equal_mask(left, right, None, Mode.PAPER).to_numpy()
    strict = equal_mask(left, right, None, Mode.STRICT).to_numpy()

    assert (strict & ~paper).sum() == 0


@settings(max_examples=25, deadline=None)
@given(
    st.lists(
        st.tuples(st.integers(-20, 20), st.integers(-20, 20), st.integers(-20, 20)),
        min_size=1,
        max_size=10,
    )
)
def test_scores_stay_between_zero_and_one_in_both_modes(rows):
    dirty = _frame([row[0] for row in rows])
    cleaned = _frame([row[1] for row in rows])
    ground_truth = _frame([row[2] for row in rows])

    for mode in (Mode.PAPER, Mode.STRICT):
        scores = Evaluator(dirty, ground_truth, None, mode).evaluate(cleaned).overall
        for section in (scores.detection_metrics, scores.correction_metrics):
            for value in (section.precision, section.recall, section.f1_score):
                assert 0.0 <= value <= 1.0


def test_strict_mode_finds_more_errors_than_paper_mode_in_a_stored_run():
    """On a real run, strict mode sees strictly more work: beers, as the thesis stored it.

    A strict score is not simply a lower paper score. Strict mode re-reads the dirty file too, so a
    cell holding "20" where the ground truth holds 20 becomes an error that was not there before —
    and a run that turned it into a number gets the credit for repairing it. What is guaranteed is
    the direction of the counts: every difference paper mode sees, strict mode sees as well.
    """
    benchmark = BENCHMARKS["beers"]
    cleaned = load_dataset(RESULTS_DIR / "beers" / "data" / "beers_cleaned.csv")
    numeric = set(benchmark.numeric_columns)

    paper = Evaluator(benchmark.dirty_path, benchmark.ground_truth_path, numeric, Mode.PAPER)
    strict = Evaluator(benchmark.dirty_path, benchmark.ground_truth_path, numeric, Mode.STRICT)

    paper_counts = paper.evaluate(cleaned).overall.detection_counts
    strict_counts = strict.evaluate(cleaned).overall.detection_counts

    assert strict_counts.total_errors > paper_counts.total_errors
    assert strict_counts.total_changes >= paper_counts.total_changes
