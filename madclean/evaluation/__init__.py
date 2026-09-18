"""Dataset-level evaluation helpers (e.g. cleaning vs ground truth)."""

from madclean.evaluation.dataset_evaluation import (
    check_frames_compatible,
    compute_cleaning_metrics,
)

__all__ = ["check_frames_compatible", "compute_cleaning_metrics"]
