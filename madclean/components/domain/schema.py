from dataclasses import dataclass, field

@dataclass
class OutlierResult:
    """Container for statistical outlier data and context for LLM."""
    outliers: list[tuple[int | float, int]]
    median: int | float
    mad: int | float
    context: list[list]

@dataclass
class FDResult:
    """Container for data about specific functional dependency."""
    lhs: str
    rhs: str
    score: float
    violations_count: int = 0
    imputables_count: int = 0
    violation_data: dict | None = None
    imputation_data: dict | None = None

@dataclass
class ColumnProfile:
    """Container for all data about a single column."""
    name: str
    semantic_type: str
    sample: str
    outlier_data: OutlierResult | None = None
    metadata: dict = field(default_factory=dict) # For future extensions

@dataclass
class MultiColumnTask:
    """Container for multi-column tasks."""
    task_type: str
    target_columns: list[str]
    verbose_key: str = ""
    data: object = None
    