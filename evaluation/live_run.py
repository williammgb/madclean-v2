"""One real-LLM benchmark run of one dataset through the thesis evaluation runner. Used by `./run live <dataset>`."""
import json
import sys
from pathlib import Path

from madclean.llm.llm_registry import LLM_CLIENT_MAP
from madclean.llm.llm_settings import LLM_CLIENT_NAME
from madclean.main import setup_llm
from madclean.utils.console import configure_console
from .evaluation_pipeline import DATASETS, EvaluationPipeline

# The run record sits beside the cleaned table, in a folder git already ignores.
RECORD_DIR = Path("data/cleaned")


def main(dataset: str) -> int:
    configure_console()
    if dataset not in DATASETS:
        print(f"live: unknown dataset '{dataset}'. Choose one of: {', '.join(DATASETS)}")
        return 2
    try:
        setup_llm(LLM_CLIENT_NAME, LLM_CLIENT_MAP)
    except (ValueError, EnvironmentError) as exc:
        print(f"live: {exc}")
        return 2

    runner = EvaluationPipeline({dataset: DATASETS[dataset]}, [], replication_count=1)
    runner.API_DELAY_SECONDS = 0
    result = runner.run(store_results=False)["my_framework"][dataset][0]
    if "error" in result:
        print(f"live {dataset}: {result['error']}")
        print(f"live {dataset}: no run record")
        return 1

    det, cor = result["detection_metrics"], result["correction_metrics"]
    tokens = result["token_usage"].get("total_tokens", 0)
    print(
        f"live {dataset}: det P {det['precision']:.3f} R {det['recall']:.3f} F1 {det['f1_score']:.3f} | "
        f"cor P {cor['precision']:.3f} R {cor['recall']:.3f} F1 {cor['f1_score']:.3f} | "
        f"{result['runtime_seconds']:.0f}s | {tokens:,} tok"
    )
    if result.get("report") is None or result.get("agent_stats") is None:
        print(f"live {dataset}: no run record")
        return 0
    _save_record(dataset, result)
    print(_rounds_line(dataset, result["agent_stats"]["per_column"]))
    return 0


def _save_record(dataset: str, result: dict) -> Path:
    """Writes the whole run record, overwriting the one an earlier run of this dataset left."""
    record = {
        "dataset": dataset,
        "scores": {key: result[key] for key in ("detection_metrics", "correction_metrics")},
        "runtime_seconds": result["runtime_seconds"],
        "token_usage": result["token_usage"],
        "agent_stats": result["agent_stats"],
        "report": result["report"],
    }
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    path = RECORD_DIR / f"{dataset}_run.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def _rounds_line(dataset: str, per_column: dict) -> str:
    """`live <dataset>: <N> extra rounds (<col> <n>, …) | <K> failed`, most rounds first."""
    extra = {col: stats["attempts"] - 1 for col, stats in per_column.items() if stats["attempts"] > 1}
    failed = sum(1 for stats in per_column.values() if stats.get("failed"))
    line = f"live {dataset}: {sum(extra.values())} extra rounds"
    if extra:
        ranked = sorted(extra.items(), key=lambda item: -item[1])
        line += " (" + ", ".join(f"{col} {n}" for col, n in ranked) + ")"
    return f"{line} | {failed} failed"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
