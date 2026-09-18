from pathlib import Path

import pytest

from madclean.components.dataprofiler.dataprofiler import DataProfiler
from madclean.components.dataprofiler.functional_dependencies import FunctionalDependencies
from madclean.components.dataprofiler.outlier_detection import OutlierDetection
from madclean.config.loader import load_default_cleaning_config
from madclean.utils.helpers import load_dataset

DATASETS_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmark_datasets"
SMALL = ["beers"]
LARGE = ["hospital", "movies", "rayyan", "tax", "adult", "restaurants"]


@pytest.fixture(scope="module")
def profiler():
    return DataProfiler(
        config=load_default_cleaning_config(),
        single_col_cleaners=[OutlierDetection()],
        multi_col_cleaners=[FunctionalDependencies()],
    )


@pytest.mark.parametrize(
    "dataset",
    SMALL + [pytest.param(name, marks=pytest.mark.slow) for name in LARGE],
)
def test_profiler_gives_every_column_a_semantic_type(profiler, dataset):
    dirty = load_dataset(DATASETS_DIR / f"{dataset}_dirty.csv")

    profiles, multi_col_tasks = profiler.analyse(dirty)

    assert list(profiles) == list(dirty.columns)
    assert all(profile.semantic_type for profile in profiles.values())
    assert all(task.task_type == "FD" for task in multi_col_tasks)
