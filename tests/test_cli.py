"""The command line: what each flag sets, how a dataset name resolves, and both commands end to end.

No test here calls a real model: cleaning runs go through the scripted fake client.
"""
import json
from pathlib import Path

import pytest

from fake_llm import FakeLLMClient
from madclean import main
from madclean.evaluation.datasets import DATA_DIR
from madclean.utils.helpers import load_dataset

ROOT = Path(__file__).resolve().parents[1]
BEERS_RUN_1 = ROOT / "evaluation" / "results" / "beers" / "data" / "beers_cleaned.csv"
BEERS_SCORES_1 = ROOT / "evaluation" / "results" / "beers" / "detailed_results" / "eval_results_1.json"


def config_for(*flags):
    return main.build_config(main.clean_parser().parse_args(["beers", *flags]))


def test_without_flags_the_settings_are_the_default_file():
    config = config_for()
    assert config.enable_validation and config.enable_multi_col_cleaning
    assert config.sample_sizes["STRING"] == {"random_sample_size": 150, "unique_sample_size": 250}
    assert config.sampling_seed is None


def test_the_ablation_flags_switch_off_the_validator_and_the_dependency_cleaning():
    config = config_for("--disable-validator", "--disable-fd")
    assert not config.enable_validation
    assert not config.enable_multi_col_cleaning


def test_sample_size_sets_both_halves_of_every_sample_except_free_text():
    config = config_for("--sample-size", "32", "--seed", "7")
    for kind, sizes in config.sample_sizes.items():
        if kind == "NLT":
            assert sizes == {"short_sample_size": 100, "long_sample_size": 20}
        else:
            assert set(sizes.values()) == {32}, kind
    assert config.sampling_seed == 7


def test_a_settings_file_is_read_and_the_flags_win_over_it(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"max_cleaning_attempts": 2, "enable_validation": True}), encoding="utf-8")
    config = config_for("--config", str(settings), "--disable-validator")
    assert config.max_cleaning_attempts == 2
    assert not config.enable_validation


def test_a_missing_settings_file_is_an_error_not_the_defaults(tmp_path):
    with pytest.raises(FileNotFoundError):
        config_for("--config", str(tmp_path / "nope.json"))


def test_nothing_in_a_command_line_run_waits_for_a_person(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "human_in_the_loop": True,
        "enable_user_validation": True,
        "validator_failure_strategy": "ask_user",
    }), encoding="utf-8")
    config = config_for("--config", str(settings))
    assert not config.human_in_the_loop and not config.enable_user_validation
    assert config.validator_failure_strategy == "accept_cleaned"


def test_one_model_for_all_agents_and_a_per_agent_override():
    args = main.clean_parser().parse_args(["beers", "--llm", "Qwen_9B", "--llm-validation", "Gemini"])
    assert main.agent_models(args) == {"recommender": "Qwen_9B", "coding": "Qwen_9B", "validation": "Gemini"}


def test_a_benchmark_name_resolves_to_its_dirty_file_and_ground_truth():
    assert main.resolve_dataset("beers") == (DATA_DIR / "beers_dirty.csv", DATA_DIR / "beers_gt.csv")


def test_a_file_path_is_taken_as_it_is_and_has_no_ground_truth():
    assert main.resolve_dataset(str(BEERS_RUN_1)) == (BEERS_RUN_1, None)


def test_an_unknown_dataset_names_the_benchmarks():
    with pytest.raises(FileNotFoundError, match="beers"):
        main.resolve_dataset("no_such_dataset")


def test_evaluate_reproduces_the_committed_score_of_a_stored_run(tmp_path, capsys):
    results = tmp_path / "scores.json"
    assert main.cli(["evaluate", str(BEERS_RUN_1), "beers", "--results", str(results)]) == 0

    committed = json.loads(BEERS_SCORES_1.read_text(encoding="utf-8"))
    written = json.loads(results.read_text(encoding="utf-8"))
    assert written["scores"] == committed
    assert set(written["per_column"]) == set(load_dataset(BEERS_RUN_1).columns)
    f1 = committed["detection_metrics"]["f1_score"]
    assert f"F1 {f1:.3f}" in capsys.readouterr().out


def test_evaluate_finds_the_dirty_file_next_to_a_ground_truth_file(tmp_path):
    results = tmp_path / "scores.json"
    gt = DATA_DIR / "beers_gt.csv"
    assert main.cli(["evaluate", str(BEERS_RUN_1), str(gt), "--results", str(results)]) == 0
    assert json.loads(results.read_text(encoding="utf-8"))["dirty"] == str(DATA_DIR / "beers_dirty.csv")


def test_evaluate_asks_for_the_dirty_file_when_it_cannot_find_one(tmp_path, capsys):
    gt = tmp_path / "truth.csv"
    gt.write_bytes((DATA_DIR / "beers_gt.csv").read_bytes())
    assert main.cli(["evaluate", str(BEERS_RUN_1), str(gt)]) == 2
    assert "--dirty" in capsys.readouterr().out


def test_an_unknown_model_is_a_configuration_error(capsys):
    assert main.cli(["beers", "--llm", "NoSuchModel"]) == 2
    assert "Invalid LLM client" in capsys.readouterr().out


def test_two_scored_runs_with_the_fake_model_write_every_run_and_the_mean(tmp_path, monkeypatch, capsys):
    columns = ["brewery_id", "brewery_name", "city"]
    dirty, gt = tmp_path / "mini_dirty.csv", tmp_path / "mini_gt.csv"
    load_dataset(DATA_DIR / "beers_dirty.csv")[columns].head(60).to_csv(dirty, index=False)
    load_dataset(DATA_DIR / "beers_gt.csv")[columns].head(60).to_csv(gt, index=False)

    fake = FakeLLMClient()
    monkeypatch.setattr(main, "LLM_CLIENT_MAP", {"Fake": fake.llm_config()})
    monkeypatch.setenv("FAKE_API_KEY", "not-a-real-key")
    results = tmp_path / "runs.json"

    code = main.cli([
        str(dirty), "--gt", str(gt), "--llm", "Fake", "--runs", "2", "--disable-validator",
        "--sample-size", "32", "--seed", "1", "--results", str(results),
    ])

    assert code == 0
    out = capsys.readouterr().out
    assert "run 1/2" in out and "run 2/2" in out and "mean of 2 runs" in out
    written = json.loads(results.read_text(encoding="utf-8"))
    assert written["models"] == {"recommender": "Fake", "coding": "Fake", "validation": "Fake"}
    assert written["settings"]["enable_validation"] is False
    assert written["settings"]["sample_sizes"]["STRING"]["random_sample_size"] == 32
    assert [run["run"] for run in written["runs"]] == [1, 2]
    assert all(run["scores"]["detection_counts"]["total_errors"] >= 0 for run in written["runs"])
    assert written["mean"]["detection_metrics"]["f1_score"]["runs"] == 2
    # With the validator off, the fake model is never asked to validate.
    assert not fake.calls_for("validator")
