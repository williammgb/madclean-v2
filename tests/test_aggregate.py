"""Averaging runs: the mean has to stay the number the paper reports, the spread is new."""

import json
import math
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from madclean.evaluation import BENCHMARKS, Evaluator, aggregate, means_only
from madclean.evaluation.aggregate import spread
from madclean.utils.helpers import load_dataset

CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_DIR / "evaluation" / "results"


def test_one_run_has_a_mean_and_no_spread():
    result = spread([0.75])
    assert result.mean == 0.75
    assert result.standard_deviation == 0.0
    assert result.runs == 1


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=1, max_size=8))
def test_the_mean_is_inside_the_range_and_the_spread_is_never_negative(values):
    result = spread(values)
    assert min(values) - 1e-12 <= result.mean <= max(values) + 1e-12
    assert result.standard_deviation >= 0.0
    # Identical runs have no spread, up to the last bits of a floating point division.
    if max(values) - min(values) < 1e-12:
        assert result.standard_deviation < 1e-9
    else:
        assert result.standard_deviation > 0.0


@settings(max_examples=30, deadline=None)
@given(st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=2, max_size=6))
def test_the_spread_is_the_population_standard_deviation(values):
    mean = sum(values) / len(values)
    expected = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    assert math.isclose(spread(values).standard_deviation, expected, abs_tol=1e-12)


@pytest.mark.parametrize("dataset", sorted(BENCHMARKS))
@pytest.mark.slow
def test_the_averages_reproduce_the_committed_average_results(dataset):
    """Four stored runs, averaged, are the numbers in avg_eval_results.json."""
    benchmark = BENCHMARKS[dataset]
    evaluator = Evaluator(
        benchmark.dirty_path, benchmark.ground_truth_path, set(benchmark.numeric_columns)
    )
    runs = []
    for run in range(1, 5):
        name = f"{dataset}_cleaned.csv" if run == 1 else f"{dataset}_cleaned_{run}.csv"
        runs.append(evaluator.evaluate(load_dataset(RESULTS_DIR / dataset / "data" / name)).overall)

    averaged = means_only(aggregate(runs))
    committed = json.loads(
        (RESULTS_DIR / dataset / "avg_eval_results.json").read_text(encoding="utf-8")
    )

    for section, values in averaged.items():
        for name, value in values.items():
            assert math.isclose(value, committed[section][name], abs_tol=5e-5), (
                f"{dataset} {section}.{name}: scored {value}, committed {committed[section][name]}"
            )


@pytest.mark.slow
def test_the_spread_is_reported_beside_every_average():
    """Every averaged field carries how far the four runs were apart."""
    benchmark = BENCHMARKS["beers"]
    evaluator = Evaluator(
        benchmark.dirty_path, benchmark.ground_truth_path, set(benchmark.numeric_columns)
    )
    runs = [
        evaluator.evaluate(
            load_dataset(
                RESULTS_DIR / "beers" / "data" / (f"beers_cleaned.csv" if run == 1 else f"beers_cleaned_{run}.csv")
            )
        ).overall
        for run in range(1, 5)
    ]

    averaged = aggregate(runs)

    for section in (averaged.detection_metrics, averaged.correction_metrics):
        for measurement in section.values():
            assert measurement.runs == 4
            assert measurement.standard_deviation >= 0.0
    # The four stored beers runs are not identical, so at least one score has to move.
    assert any(m.standard_deviation > 0 for m in averaged.detection_metrics.values())
