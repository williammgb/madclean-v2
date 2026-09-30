import os
import sys
import copy
import json
import argparse
import subprocess
from dataclasses import asdict
from pathlib import Path
from dotenv import load_dotenv
### Local imports
from madclean.pipeline import Pipeline
from madclean.config.settings import CleaningConfig
from madclean.config.loader import load_default_cleaning_config
from madclean.evaluation import BENCHMARKS, Evaluator, Mode, aggregate
from madclean.evaluation.datasets import DATA_DIR
from madclean.llm.llm_settings import LLM_CLIENT_NAME
from madclean.llm.llm_registry import LLM_CLIENT_MAP, LLMSpec
from madclean.notebook import notebook_json
from madclean.utils.console import configure_console
from madclean.utils.helpers import save_dataset

REPO_DIR = Path(__file__).resolve().parents[1]
AGENTS = ("recommender", "coding", "validation")

def run_ui() -> int:
    """Launch the Reflex UI from the repository GUI folder."""
    gui_dir = REPO_DIR / "gui"
    if not gui_dir.exists():
        raise FileNotFoundError(f"Could not find GUI directory at: {gui_dir}")
    try:
        subprocess.run(["reflex", "run"], cwd=str(gui_dir), check=True)
        return 0
    except FileNotFoundError as ex:
        raise RuntimeError("`reflex` is not installed or not on PATH.") from ex

def cli_ui() -> int:
    """Dedicated UI entry point for pyproject console script."""
    configure_console()
    return run_ui()

def setup_llm(llm_client_name: str, llm_clients: dict[str, LLMSpec]) -> LLMSpec:
    load_dotenv()
    if llm_client_name not in llm_clients:
        raise ValueError(
            f"Invalid LLM client '{llm_client_name}'. "
            f"Available options: {list(llm_clients.keys())} \n"
            f"Or view README to add LLM API."
        )
    llm_config = llm_clients[llm_client_name]
    api_key_name = llm_config.api_key_name
    if not api_key_name:
        raise ValueError(
            f"api_key_name not defined in registry for {llm_client_name}"
            f"First add it before running the pipeline."
        )
    api_key = os.getenv(api_key_name)
    if not api_key:
        raise EnvironmentError(
            f"Missing API Key: {api_key_name} not found in .env file. "
            f"First add it before running the pipeline."
        )
    return llm_config

### Datasets

def resolve_dataset(name_or_path: str) -> tuple[Path, Path | None]:
    """A dataset argument as (dirty file, ground truth or None).

    An existing file is taken as it is. Otherwise a benchmark name such as `beers` stands for
    `data/benchmark_datasets/beers_dirty.csv`, and its `_gt.csv` next to it is the ground truth.
    """
    path = Path(name_or_path)
    if path.is_file():
        return path, None
    dirty = DATA_DIR / f"{name_or_path}_dirty.csv"
    if dirty.is_file():
        gt = DATA_DIR / f"{name_or_path}_gt.csv"
        return dirty, gt if gt.is_file() else None
    names = sorted(p.name[: -len("_dirty.csv")] for p in DATA_DIR.glob("*_dirty.csv"))
    raise FileNotFoundError(f"'{name_or_path}' is neither a file nor a benchmark. Benchmarks: {', '.join(names)}")

def numeric_columns_for(ground_truth: Path) -> set[str] | None:
    """The columns a benchmark declares numeric, found by its ground truth's file name.

    Any other file has no declared schema, and None makes the scorer compare every column as
    numbers where it can.
    """
    for bench in BENCHMARKS.values():
        if ground_truth.name == bench.ground_truth_path.name:
            return set(bench.numeric_columns)
    return None

def dirty_next_to(ground_truth: Path) -> Path | None:
    """`X_dirty.ext` beside `X_gt.ext`, the benchmark naming, when it exists."""
    if not ground_truth.stem.endswith("_gt"):
        return None
    dirty = ground_truth.with_name(ground_truth.stem[: -len("_gt")] + "_dirty" + ground_truth.suffix)
    return dirty if dirty.is_file() else None

### Settings

def set_sample_size(config: CleaningConfig, size: int) -> None:
    """One sample size for both halves of every column type's sample, as in the paper's sample-size study.

    Free-text columns (NLT) keep their own sizes: their two halves are short and long texts, not
    random and dirty values.
    """
    for kind, sizes in config.sample_sizes.items():
        if kind == "NLT":
            continue
        for key in sizes:
            sizes[key] = size

def build_config(args: argparse.Namespace) -> CleaningConfig:
    """The run's settings: the JSON file (or the default configurations.json), then the flags on top."""
    if args.config and not Path(args.config).is_file():
        raise FileNotFoundError(f"Settings file not found: {args.config}")
    config = load_default_cleaning_config(args.config)
    config.verbose = args.verbose
    if args.disable_validator:
        config.enable_validation = False
    if args.disable_fd:
        config.enable_multi_col_cleaning = False
    if args.sample_size is not None:
        set_sample_size(config, args.sample_size)
    if args.seed is not None:
        config.sampling_seed = args.seed
    # Nobody is at a prompt to review a step, so no step may wait for a person.
    config.human_in_the_loop = False
    config.enable_user_validation = False
    if config.validator_failure_strategy == "ask_user":
        config.validator_failure_strategy = "accept_cleaned"
    return config

def agent_models(args: argparse.Namespace) -> dict[str, str]:
    """The model key for each agent: its own flag, else --llm, else LLM_CLIENT_NAME."""
    default = args.llm or LLM_CLIENT_NAME
    return {agent: getattr(args, f"llm_{agent}") or default for agent in AGENTS}

### Output

def score_line(scores) -> str:
    det, cor = scores.detection_metrics, scores.correction_metrics
    return (
        f"detection P {det.precision:.3f} R {det.recall:.3f} F1 {det.f1_score:.3f} | "
        f"correction P {cor.precision:.3f} R {cor.recall:.3f} F1 {cor.f1_score:.3f}"
    )

def mean_line(runs: list) -> str:
    mean = aggregate(runs)
    parts = []
    for label, section in (("detection", mean.detection_metrics), ("correction", mean.correction_metrics)):
        parts.append(" ".join(
            f"{name} {section[key].mean:.3f} ± {section[key].standard_deviation:.3f}"
            for name, key in (("P", "precision"), ("R", "recall"), ("F1", "f1_score"))
        ))
    return f"mean of {len(runs)} runs: detection {parts[0]} | correction {parts[1]}"

def write_json(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results written to {path}")

def numbered(path: str, run: int, runs: int) -> Path:
    """The path itself for a single run; `name_run2.ext` and so on when there are several."""
    path = Path(path)
    return path if runs == 1 else path.with_name(f"{path.stem}_run{run}{path.suffix}")

### Commands

def clean(args: argparse.Namespace) -> int:
    """Cleans one dataset, `--runs` times, scoring each run when a ground truth is known."""
    try:
        dirty_path, ground_truth = resolve_dataset(args.dataset)
        if args.gt:
            ground_truth = Path(args.gt)
            if not ground_truth.is_file():
                raise FileNotFoundError(f"Ground truth not found: {args.gt}")
        config = build_config(args)
        models = agent_models(args)
        llm_configs = {agent: setup_llm(key, LLM_CLIENT_MAP) for agent, key in models.items()}
    except (ValueError, EnvironmentError) as e:
        print(f"Configuration Error: {e}")
        return 2

    evaluator = None
    if ground_truth is not None:
        try:
            evaluator = Evaluator(dirty_path, ground_truth, numeric_columns_for(ground_truth), args.mode)
        except ValueError as e:
            print(f"Cannot score against {ground_truth}: {e}")
            return 2

    print(f"Cleaning {dirty_path} with {', '.join(f'{a}={m}' for a, m in models.items())}"
          + (f", scored against {ground_truth} ({args.mode} mode)" if evaluator else ""))
    results, scored = [], []
    for run in range(1, args.runs + 1):
        pipeline = Pipeline(
            llm_config=llm_configs["coding"],
            agent_llm_configs=llm_configs,
            config=copy.deepcopy(config),
            verbose=args.verbose,
        )
        cleaned, report = pipeline.run(file_path=str(dirty_path))
        if cleaned is None or report is None:
            print(f"run {run}/{args.runs}: the pipeline returned no cleaned table")
            return 1
        result = {
            "run": run,
            "runtime_seconds": report.runtime_seconds,
            "tokens": asdict(report.total_usage),
            "tokens_per_agent": asdict(report.token_usage),
            "cleaned_file": None,
            "scores": None,
            "per_column": None,
        }
        line = f"run {run}/{args.runs}: {report.runtime_seconds:.0f}s, {report.total_usage.total_tokens:,} tokens"
        if evaluator is not None:
            evaluation = evaluator.evaluate(cleaned)
            scored.append(evaluation.overall)
            result["scores"] = asdict(evaluation.overall)
            result["per_column"] = {col: asdict(s) for col, s in evaluation.per_column.items()}
            line += f" | {score_line(evaluation.overall)}"
        if args.save_cleaned:
            saved = save_dataset(cleaned, str(dirty_path), REPO_DIR)
            result["cleaned_file"] = str(saved)
            line += f" | saved {saved}"
        if args.export_notebook:
            # The run as a notebook: the agents' code, in the order it ran, with no model call left.
            notebook = numbered(args.export_notebook, run, args.runs)
            output = result["cleaned_file"] or str(dirty_path.with_name(dirty_path.stem + "_cleaned.csv"))
            notebook.write_text(
                notebook_json(report, dataset_path=str(dirty_path), output_path=output),
                encoding="utf-8",
            )
            line += f" | notebook {notebook}"
        print(line)
        results.append(result)

    if len(scored) > 1:
        print(mean_line(scored))
    if args.results:
        write_json(args.results, {
            "dataset": str(dirty_path),
            "ground_truth": str(ground_truth) if ground_truth else None,
            "scoring_mode": args.mode if evaluator else None,
            "models": models,
            "settings": asdict(config),
            "runs": results,
            "mean": asdict(aggregate(scored)) if scored else None,
        })
    return 0

def evaluate(args: argparse.Namespace) -> int:
    """Scores a cleaned file that already exists against its ground truth."""
    try:
        cleaned = Path(args.cleaned)
        if not cleaned.is_file():
            raise FileNotFoundError(f"Cleaned file not found: {args.cleaned}")
        if Path(args.ground_truth).is_file():
            ground_truth = Path(args.ground_truth)
            dirty = Path(args.dirty) if args.dirty else dirty_next_to(ground_truth)
        else:
            dirty, ground_truth = resolve_dataset(args.ground_truth)
            dirty = Path(args.dirty) if args.dirty else dirty
            if ground_truth is None:
                raise FileNotFoundError(f"Benchmark '{args.ground_truth}' has no ground truth file.")
        if dirty is None:
            raise FileNotFoundError(
                "Scoring needs the dirty file too, to know which cells were errors: pass --dirty PATH."
            )
        if not dirty.is_file():
            raise FileNotFoundError(f"Dirty file not found: {dirty}")
        evaluator = Evaluator(dirty, ground_truth, numeric_columns_for(ground_truth), args.mode)
        evaluation = evaluator.evaluate(cleaned)
    except (ValueError, OSError) as e:
        print(f"Evaluation Error: {e}")
        return 2

    counts = evaluation.overall.detection_counts
    repaired = evaluation.overall.correction_counts.correctly_repaired_cells
    print(f"Scored {cleaned} against {ground_truth} ({args.mode} mode), dirty file {dirty}")
    print(score_line(evaluation.overall))
    print(f"errors in the dirty file {counts.total_errors:,} | cells changed {counts.total_changes:,} "
          f"| correctly repaired {repaired:,}")
    if args.results:
        write_json(args.results, {
            "cleaned": str(cleaned),
            "ground_truth": str(ground_truth),
            "dirty": str(dirty),
            "scoring_mode": args.mode,
            "scores": asdict(evaluation.overall),
            "per_column": {col: asdict(s) for col, s in evaluation.per_column.items()},
        })
    return 0

### Parsers

def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value

def clean_parser() -> argparse.ArgumentParser:
    models = ", ".join(LLM_CLIENT_MAP)
    parser = argparse.ArgumentParser(
        prog="madclean",
        description="Clean a dataset with the multi-agent pipeline. "
                    "Also: `madclean evaluate --help` to score a cleaned file, `madclean ui` for the web interface.",
    )
    parser.add_argument(
        "dataset",
        help="A CSV/Excel/JSON file, or a benchmark name (beers, hospital, movies, rayyan, tax, adult, restaurants). "
             "A benchmark is scored against its ground truth automatically.",
    )
    models_group = parser.add_argument_group("models", f"Model keys: {models}. Default: {LLM_CLIENT_NAME}.")
    models_group.add_argument("--llm", metavar="MODEL", help="One model for all three agents.")
    models_group.add_argument("--llm-recommender", metavar="MODEL", help="Model for the Recommender agent.")
    models_group.add_argument("--llm-coding", metavar="MODEL", help="Model for the Coder agent.")
    models_group.add_argument("--llm-validation", metavar="MODEL", help="Model for the Validator agent.")

    settings = parser.add_argument_group("settings")
    settings.add_argument("--config", metavar="PATH",
                          help="JSON settings file with the same keys as configurations.json (the default). "
                               "The flags below override it.")
    settings.add_argument("--disable-validator", action="store_true",
                          help="Skip the Validator agent: the Coder's output is applied unchecked.")
    settings.add_argument("--disable-fd", action="store_true",
                          help="Skip cleaning with functional dependencies (rules between columns).")
    settings.add_argument("--sample-size", type=positive_int, metavar="N",
                          help="Rows the agents see per column: N random and N dirty values "
                               "(free-text columns keep their own sizes).")
    settings.add_argument("--seed", type=int, metavar="N",
                          help="Makes the samples the agents see the same on every run.")

    output = parser.add_argument_group("runs and output")
    output.add_argument("--runs", type=positive_int, default=1, metavar="N",
                        help="Clean the dataset N times; scores are printed per run and as a mean.")
    output.add_argument("--gt", metavar="PATH", help="Ground truth file to score each run against.")
    output.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.PAPER.value,
                        help="Scoring mode (default: paper).")
    output.add_argument("--save-cleaned", action="store_true",
                        help="Write each cleaned table to data/cleaned/.")
    output.add_argument("--results", metavar="PATH",
                        help="Write settings, runtime, tokens and scores of every run to a JSON file.")
    output.add_argument("--export-notebook", metavar="PATH",
                        help="Write the run as a notebook (.ipynb) that reproduces it without calling a model.")
    output.add_argument("-v", "--verbose", action="store_true", help="Print profiling and per-agent progress.")
    return parser

def evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="madclean evaluate",
        description="Score a cleaned file against its ground truth. No model is called.",
    )
    parser.add_argument("cleaned", help="The cleaned file to score.")
    parser.add_argument("ground_truth",
                        help="The ground truth file, or a benchmark name such as beers.")
    parser.add_argument("--dirty", metavar="PATH",
                        help="The dirty file the cleaned one came from. "
                             "Found automatically for a benchmark and for X_gt.csv next to X_dirty.csv.")
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.PAPER.value,
                        help="Scoring mode (default: paper).")
    parser.add_argument("--results", metavar="PATH", help="Write the scores, overall and per column, to a JSON file.")
    return parser

def cli(argv=None) -> int:
    """CLI wrapper. Returns a process exit code."""
    configure_console()
    argv = sys.argv[1:] if argv is None else list(argv)
    command = argv[0].lower() if argv else ""
    if command == "ui":
        return run_ui()
    if command == "evaluate":
        return evaluate(evaluate_parser().parse_args(argv[1:]))
    return clean(clean_parser().parse_args(argv))

if __name__ == "__main__":
    raise SystemExit(cli())
