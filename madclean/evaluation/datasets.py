"""The benchmark datasets and what each one declares numeric.

The numeric columns matter: in paper mode they are the columns compared as numbers, so "20" and 20
count as the same value there and nowhere else. The sets are the thesis's, unchanged, because the
committed results were produced with them.

Only the four datasets the paper reports are listed. tax, adult and restaurants are in `data/` and
can be cleaned, but no stored results exist to compare them against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmark_datasets"


@dataclass(frozen=True)
class Benchmark:
    """One dataset a cleaning run can be scored on."""
    name: str
    numeric_columns: frozenset[str]

    @property
    def dirty_path(self) -> Path:
        return DATA_DIR / f"{self.name}_dirty.csv"

    @property
    def ground_truth_path(self) -> Path:
        return DATA_DIR / f"{self.name}_gt.csv"


BENCHMARKS: dict[str, Benchmark] = {
    "hospital": Benchmark(
        "hospital",
        frozenset({"ProviderNumber", "ZipCode", "PhoneNumber", "Score", "Sample"}),
    ),
    "beers": Benchmark(
        "beers",
        frozenset({"id", "ounces", "abv", "ibu", "brewery_id"}),
    ),
    "movies": Benchmark(
        "movies",
        frozenset({"Year", "Duration", "RatingValue", "RatingCount"}),
    ),
    "rayyan": Benchmark(
        "rayyan",
        frozenset({"id", "article_jvolumn", "article_jissue"}),
    ),
}


def benchmark(name: str) -> Benchmark:
    """The benchmark by name, with a message that lists the choices when there is no such one."""
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise KeyError(f"unknown dataset '{name}'. Choose one of: {', '.join(BENCHMARKS)}") from None


def as_paths(name: str) -> dict[str, object]:
    """The dictionary shape the benchmark scripts pass around."""
    found = benchmark(name)
    return {
        "dirty_path": found.dirty_path,
        "ground_truth_path": found.ground_truth_path,
        "numeric_cols": set(found.numeric_columns),
    }
