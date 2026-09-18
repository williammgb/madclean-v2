"""Evaluation: one scorer, used by the command line, the GUI and the benchmark scripts.

`Evaluator` scores a cleaned dataset against its ground truth in one of two modes (see
`comparison.Mode`); `aggregate` averages runs; `agent_stats` says what the run cost. The GUI's
metric names live in `gui_metrics` and are re-exported here, because that is where the GUI has
always imported them from.
"""

from madclean.evaluation.agent_stats import agent_stats
from madclean.evaluation.aggregate import aggregate, aggregate_per_column, means_only
from madclean.evaluation.comparison import Mode, equal_mask
from madclean.evaluation.datasets import BENCHMARKS, Benchmark, benchmark
from madclean.evaluation.gui_metrics import (
    check_frames_compatible,
    compute_cleaning_metrics,
    format_int,
    format_pct,
)
from madclean.evaluation.scoring import Evaluator

__all__ = [
    "BENCHMARKS",
    "Benchmark",
    "Evaluator",
    "Mode",
    "agent_stats",
    "aggregate",
    "aggregate_per_column",
    "benchmark",
    "check_frames_compatible",
    "compute_cleaning_metrics",
    "equal_mask",
    "format_int",
    "format_pct",
    "means_only",
]
