"""One command that re-scores everything this repository has stored.

Every cleaned file under `evaluation/results/` (MADClean) and `evaluation/baselines/*/data/`
(the systems it is compared against) is scored again from disk, in both modes, and printed as one
table. Nothing is cleaned and no model is called, so the command is free, offline and repeatable:
it answers "what do the stored outputs score today, with today's scorer".

Run it with `./run score`, or `python -m madclean.evaluation.scoreboard`.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from madclean.evaluation.aggregate import aggregate
from madclean.evaluation.comparison import Mode
from madclean.evaluation.datasets import BENCHMARKS, Benchmark
from madclean.evaluation.scoring import Evaluator
from madclean.utils.console import configure_console

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
BASELINES_DIR = REPO_ROOT / "evaluation" / "baselines"
SCOREBOARD_PATH = RESULTS_DIR / "scoreboard.json"

# beers_cleaned.csv is run 1, beers_cleaned_2.csv is run 2, and so on.
_RUN_NUMBER = re.compile(r"_cleaned(?:_(\d+))?\.csv$")


def cleaned_files(method: str, dataset: str) -> list[Path]:
    """Every stored cleaned file for one method and dataset, in run order."""
    if method == "madclean":
        folder = RESULTS_DIR / dataset / "data"
    else:
        folder = BASELINES_DIR / method / "data"
    if not folder.is_dir():
        return []
    found = sorted(folder.glob(f"{dataset}_cleaned*.csv"), key=_run_number)
    return [path for path in found if _RUN_NUMBER.search(path.name)]


def _run_number(path: Path) -> int:
    match = _RUN_NUMBER.search(path.name)
    return int(match.group(1) or 1) if match else 0


def methods() -> list[str]:
    """MADClean first, then every baseline that has stored output, in alphabetical order."""
    baselines = sorted(folder.name for folder in BASELINES_DIR.iterdir() if (folder / "data").is_dir())
    return ["madclean"] + baselines


def score_method(method: str, benchmark: Benchmark, mode: Mode) -> dict | None:
    """Scores every stored run of one method on one dataset, averaged. None when it has none."""
    files = cleaned_files(method, benchmark.name)
    if not files:
        return None
    evaluator = Evaluator(
        benchmark.dirty_path,
        benchmark.ground_truth_path,
        numeric_columns=set(benchmark.numeric_columns),
        mode=mode,
    )
    runs = []
    for path in files:
        try:
            runs.append(evaluator.evaluate(path).overall)
        except ValueError as exc:
            print(f"  {method:<12} {benchmark.name:<9} skipped {path.name}: {exc}")
    if not runs:
        return None
    averaged = aggregate(runs)
    return {
        "runs": len(runs),
        "detection": asdict(averaged.detection_metrics["f1_score"]),
        "correction": asdict(averaged.correction_metrics["f1_score"]),
        "detection_precision": averaged.detection_metrics["precision"].mean,
        "detection_recall": averaged.detection_metrics["recall"].mean,
        "correction_precision": averaged.correction_metrics["precision"].mean,
        "correction_recall": averaged.correction_metrics["recall"].mean,
    }


def build() -> dict:
    """Scores every method on every benchmark, in both modes."""
    board: dict = {}
    for mode in (Mode.PAPER, Mode.STRICT):
        board[mode.value] = {}
        for name, benchmark in BENCHMARKS.items():
            scored = {}
            for method in methods():
                result = score_method(method, benchmark, mode)
                if result is not None:
                    scored[method] = result
            board[mode.value][name] = scored
    return board


def print_board(board: dict) -> None:
    """Prints the scoreboard as one table per mode: detection F1 and correction F1, mean ± sd."""
    for mode, datasets in board.items():
        print(f"\n{mode.upper()} MODE — detection F1 / correction F1, mean ± standard deviation")
        print("-" * 78)
        print(f"{'method':<14}" + "".join(f"{name:<16}" for name in datasets))
        for method in methods():
            if not any(method in scored for scored in datasets.values()):
                continue
            line = f"{method:<14}"
            for scored in datasets.values():
                result = scored.get(method)
                if result is None:
                    line += f"{'—':<16}"
                else:
                    line += f"{result['detection']['mean']:.3f}/{result['correction']['mean']:.3f}   "
            print(line)
        print()
        for name, scored in datasets.items():
            madclean = scored.get("madclean")
            if madclean and madclean["runs"] > 1:
                detection, correction = madclean["detection"], madclean["correction"]
                print(
                    f"  madclean {name:<9} over {madclean['runs']} runs: "
                    f"detection {detection['mean']:.4f} ± {detection['standard_deviation']:.4f}, "
                    f"correction {correction['mean']:.4f} ± {correction['standard_deviation']:.4f}"
                )


def main(argv: list[str] | None = None) -> int:
    """Scores everything stored and prints it. There is nothing to configure, so argv is ignored."""
    configure_console()
    board = build()
    print_board(board)
    SCOREBOARD_PATH.write_text(json.dumps(board, indent=2), encoding="utf-8")
    print(f"\nWritten to {_short(SCOREBOARD_PATH)}")
    return 0


def _short(path: Path) -> str:
    """The path as the reader knows it: relative to the repository when it is inside it."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
