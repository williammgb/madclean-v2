"""The records evaluation produces: the scores of one run, of many runs, and of the agents.

`Scores` keeps the field names and the nesting of the committed results JSON, so `asdict` of a
`Scores` is exactly the dictionary the thesis scorer returned. Everything added here sits beside
it rather than inside it, so the stored results stay readable by the code that wrote them.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pandas is only needed for the type of the detailed reports
    import pandas as pd


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


@dataclass
class RunEvaluation:
    """What scoring one cleaned dataset produced: the table, its columns, and what changed where.

    `reports` holds one frame per column that had errors or changes, with the dirty, cleaned and
    ground truth value side by side and how that cell turned out. It is for reading, not counting.
    """
    overall: Scores
    per_column: dict[str, Scores]
    mode: str
    reports: "dict[str, pd.DataFrame]" = field(default_factory=dict)


@dataclass
class Spread:
    """One number measured over several runs: what it averaged, and how much it moved."""
    mean: float
    standard_deviation: float
    runs: int


@dataclass
class AggregateScores:
    """The same scores as `Scores`, averaged over runs, each with its spread."""
    detection_counts: dict[str, Spread]
    detection_metrics: dict[str, Spread]
    correction_counts: dict[str, Spread]
    correction_metrics: dict[str, Spread]


@dataclass
class ColumnAgentStats:
    """How much work the agents did on one column."""
    attempts: int = 0
    validator_rejections: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    already_clean: bool = False
    cleaned: bool = False
    validated: bool = False
    failed: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class AgentStats:
    """How much work one run cost, per column and in total."""
    per_column: dict[str, ColumnAgentStats] = field(default_factory=dict)
    runtime_seconds: float | None = None

    @property
    def columns(self) -> int:
        return len(self.per_column)

    @property
    def attempts(self) -> int:
        return sum(column.attempts for column in self.per_column.values())

    @property
    def validator_rejections(self) -> int:
        return sum(column.validator_rejections for column in self.per_column.values())

    @property
    def total_tokens(self) -> int:
        return sum(column.total_tokens for column in self.per_column.values())
