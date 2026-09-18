"""One real-LLM benchmark run of one dataset through the thesis evaluation runner. Used by `./run live <dataset>`."""
import sys

from madclean.llm.llm_registry import LLM_CLIENT_MAP
from madclean.llm.llm_settings import LLM_CLIENT_NAME
from madclean.main import setup_llm
from madclean.utils.console import configure_console
from .evaluation_pipeline import DATASETS, EvaluationPipeline


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
        return 1

    det, cor = result["detection_metrics"], result["correction_metrics"]
    tokens = result["token_usage"].get("total_tokens", 0)
    print(
        f"live {dataset}: det P {det['precision']:.3f} R {det['recall']:.3f} F1 {det['f1_score']:.3f} | "
        f"cor P {cor['precision']:.3f} R {cor['recall']:.3f} F1 {cor['f1_score']:.3f} | "
        f"{result['runtime_seconds']:.0f}s | {tokens:,} tok"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
