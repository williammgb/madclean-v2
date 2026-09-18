import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from madclean.evaluation import BENCHMARKS, Evaluator
from madclean.utils.helpers import load_dataset

CODE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = CODE_DIR / "data" / "benchmark_datasets"
RESULTS_DIR = CODE_DIR / "evaluation" / "results"

# The numeric columns the thesis scored with now live with the scorer; these are those sets.
NUMERIC_COLUMNS = {name: set(found.numeric_columns) for name, found in BENCHMARKS.items()}


def _stored_runs():
    for dataset in NUMERIC_COLUMNS:
        for run in range(1, 5):
            marks = [] if (dataset, run) == ("beers", 1) else [pytest.mark.slow]
            yield pytest.param(dataset, run, marks=marks, id=f"{dataset}-run{run}")


@pytest.fixture(scope="module")
def evaluator_for():
    cache = {}

    def get(dataset):
        if dataset not in cache:
            cache[dataset] = Evaluator(
                DATASETS_DIR / f"{dataset}_dirty.csv",
                DATASETS_DIR / f"{dataset}_gt.csv",
                NUMERIC_COLUMNS[dataset],
            )
        return cache[dataset]

    return get


def _mismatches(actual, expected, prefix=""):
    found = []
    for key, value in expected.items():
        if isinstance(value, dict):
            found += _mismatches(actual[key], value, f"{prefix}{key}.")
        elif not math.isclose(actual[key], value, abs_tol=5e-5):
            found.append(f"{prefix}{key}: got {actual[key]}, committed {value}")
    return found


@pytest.mark.parametrize(("dataset", "run"), list(_stored_runs()))
def test_stored_cleaned_output_reproduces_committed_scores(evaluator_for, dataset, run):
    name = f"{dataset}_cleaned.csv" if run == 1 else f"{dataset}_cleaned_{run}.csv"
    cleaned = load_dataset(RESULTS_DIR / dataset / "data" / name)
    evaluation = evaluator_for(dataset).evaluate(cleaned)
    overall, per_column = evaluation.overall, evaluation.per_column

    details = RESULTS_DIR / dataset / "detailed_results"
    committed_overall = json.loads((details / f"eval_results_{run}.json").read_text(encoding="utf-8"))
    committed_columns = json.loads((details / f"col_results_{run}.json").read_text(encoding="utf-8"))

    assert set(committed_overall) == {"detection_counts", "detection_metrics", "correction_counts", "correction_metrics"}
    assert set(committed_columns) == set(per_column)
    assert _mismatches(asdict(overall), committed_overall) == []
    assert _mismatches({col: asdict(scores) for col, scores in per_column.items()}, committed_columns) == []
