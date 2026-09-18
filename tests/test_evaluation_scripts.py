import json
from pathlib import Path

import pandas as pd

from evaluation import evaluate_files, evaluation_pipeline
from fake_llm import FakeLLMClient

CODE_DIR = Path(__file__).resolve().parents[1]
BEERS = evaluation_pipeline.DATASETS["beers"]


def test_evaluate_files_reproduces_the_committed_beers_scores_and_rescores_a_baseline():
    runner = evaluate_files.EvaluationPipeline({"beers": BEERS}, ["holoclean"], replication_count=1, verbose=False)

    results = runner.run(store_results=False)

    committed = json.loads((CODE_DIR / "evaluation" / "results" / "beers" / "avg_eval_results.json").read_text(encoding="utf-8"))
    assert results["my_framework"]["beers"] == [committed]
    (holoclean,) = results["holoclean"]["beers"]
    assert "error" not in holoclean
    assert set(holoclean["detection_metrics"]) == {"precision", "recall", "f1_score"}


def test_evaluation_pipeline_runs_the_framework_and_scores_it(tmp_path, monkeypatch):
    columns = ["id", "beer_name", "abv"]
    pd.read_csv(BEERS["dirty_path"]).head(20)[columns].to_csv(tmp_path / "slice_dirty.csv", index=False)
    pd.read_csv(BEERS["ground_truth_path"]).head(20)[columns].to_csv(tmp_path / "slice_gt.csv", index=False)
    fake = FakeLLMClient()
    monkeypatch.setitem(evaluation_pipeline.LLM_CLIENT_MAP, evaluation_pipeline.LLM_CLIENT_NAME, fake.llm_config())
    monkeypatch.setattr(evaluation_pipeline.EvaluationPipeline, "API_DELAY_SECONDS", 0)
    monkeypatch.setattr("madclean.pipeline.save_dataset", lambda df, path, base_dir: tmp_path / "unused.csv")
    datasets = {"slice": {
        "dirty_path": tmp_path / "slice_dirty.csv",
        "ground_truth_path": tmp_path / "slice_gt.csv",
        "numeric_cols": {"id", "abv"},
    }}

    results = evaluation_pipeline.EvaluationPipeline(datasets, [], replication_count=1, verbose=False).run(store_results=False)

    (result,) = results["my_framework"]["slice"]
    assert "error" not in result
    assert set(result["detection_metrics"]) == {"precision", "recall", "f1_score"}
    assert set(result["correction_metrics"]) == {"precision", "recall", "f1_score"}
    assert set(result["token_usage"]) == {"input_tokens", "output_tokens", "total_tokens"}
    assert result["token_usage"]["total_tokens"] > 0
    assert result["runtime_seconds"] > 0
    assert fake.calls_for("coder")
