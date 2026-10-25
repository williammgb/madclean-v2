import json
from pathlib import Path

import pandas as pd

from evaluation import evaluate_files, evaluation_pipeline, live_run
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
    assert "report" in result and "agent_stats" in result
    assert set(result["agent_stats"]["per_column"]) <= set(result["report"])
    assert all(stats["attempts"] >= 1 for stats in result["agent_stats"]["per_column"].values())
    json.dumps({"report": result["report"], "agent_stats": result["agent_stats"]})


def _live_result(per_column: dict) -> dict:
    """What `run_framework` hands back for one dataset, with the given per-column agent stats."""
    scores = {"precision": 0.5, "recall": 0.5, "f1_score": 0.5}
    return {
        "detection_metrics": scores,
        "correction_metrics": scores,
        "token_usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "runtime_seconds": 3.0,
        "report": {name: {"attempts": stats["attempts"]} for name, stats in per_column.items()},
        "agent_stats": {"per_column": per_column, "runtime_seconds": 3.0},
    }


def _run_live(monkeypatch, tmp_path, result: dict) -> int:
    class FakeRunner:
        API_DELAY_SECONDS = 60

        def __init__(self, datasets, baselines, replication_count=1):
            self.dataset = next(iter(datasets))

        def run(self, store_results=True):
            return {"my_framework": {self.dataset: [result]}}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(live_run, "EvaluationPipeline", FakeRunner)
    monkeypatch.setattr(live_run, "setup_llm", lambda name, client_map: None)
    monkeypatch.setattr(live_run, "configure_console", lambda: None)
    return live_run.main("beers")


def _column(attempts: int, failed: bool = False) -> dict:
    return {"attempts": attempts, "validator_rejections": attempts - 1, "failed": failed}


def test_live_run_saves_the_run_record_and_counts_extra_rounds(tmp_path, monkeypatch, capsys):
    first = _live_result({"city": _column(2), "state": _column(1)})
    assert _run_live(monkeypatch, tmp_path, first) == 0
    capsys.readouterr()

    second = _live_result({"city": _column(3), "zip": _column(4, failed=True), "state": _column(1)})
    assert _run_live(monkeypatch, tmp_path, second) == 0

    out = capsys.readouterr().out
    assert "live beers: 5 extra rounds (zip 3, city 2) | 1 failed" in out
    record = json.loads((tmp_path / "data" / "cleaned" / "beers_run.json").read_text(encoding="utf-8"))
    assert record["agent_stats"]["per_column"]["city"]["attempts"] == 3
    assert set(record) == {"dataset", "scores", "runtime_seconds", "token_usage", "agent_stats", "report"}
    assert record["dataset"] == "beers"


def test_live_run_without_extra_rounds_prints_no_list(tmp_path, monkeypatch, capsys):
    assert _run_live(monkeypatch, tmp_path, _live_result({"city": _column(1)})) == 0

    assert "live beers: 0 extra rounds | 0 failed" in capsys.readouterr().out


def test_live_run_failed_pipeline_writes_no_record(tmp_path, monkeypatch, capsys):
    failed = evaluation_pipeline.EvaluationPipeline._get_empty_metrics(error_msg="Pipeline returned None")

    assert _run_live(monkeypatch, tmp_path, failed) == 1

    assert "live beers: no run record" in capsys.readouterr().out
    assert not (tmp_path / "data").exists()
