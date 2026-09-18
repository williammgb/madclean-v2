"""The scores one evaluation run produces.

The field names and the nesting are the ones in the committed results JSON, so `asdict` of a
`Scores` is exactly the dictionary the thesis scorer returned.
"""

from dataclasses import dataclass


@dataclass
class DetectionCounts:
    """How many cells the cleaning touched, against how many were actually wrong."""
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    total_errors: int
    total_changes: int


@dataclass
class PrecisionRecallF1:
    """One set of scores. 0.0 stands for undefined here, as in the paper."""
    precision: float
    recall: float
    f1_score: float


@dataclass
class CorrectionCounts:
    """How the touched cells turned out against the ground truth."""
    correctly_repaired_cells: int
    incorrectly_repaired_cells: int
    repaired_clean_cells: int


@dataclass
class Scores:
    """Detection and correction, counted and scored."""
    detection_counts: DetectionCounts
    detection_metrics: PrecisionRecallF1
    correction_counts: CorrectionCounts
    correction_metrics: PrecisionRecallF1
