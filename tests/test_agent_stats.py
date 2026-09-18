"""What a run cost: attempts, validator rejections and tokens, per column.

The numbers are read off the report a run already produced, so the test drives a real pipeline run
against the scripted fake model and then checks that the statistics say what the run did.
"""

import json

import pandas as pd

from fake_llm import FakeLLMClient
from madclean.config.loader import load_default_cleaning_config
from madclean.evaluation import agent_stats
from madclean.pipeline import Pipeline


def _run(tmp_path, script=None):
    frame = pd.DataFrame(
        {
            "letters": ["anna ", "Bram", "cees", "Dirk ", "eva", "Femke", "gijs", "Hanna"],
            "numbers": [10, 11, 12, 13, 14, 15, 16, 17],
        }
    )
    path = tmp_path / "small_dirty.csv"
    frame.to_csv(path, index=False)

    fake = FakeLLMClient(script=script) if script else FakeLLMClient()
    config = load_default_cleaning_config()
    config.verbose = False
    config.sampling_seed = 7
    config.enable_multi_col_cleaning = False
    llm = fake.llm_config()
    pipeline = Pipeline(
        llm_config=llm,
        agent_llm_configs={"recommender": llm, "coding": llm, "validation": llm},
        config=config,
        verbose=False,
    )
    _, report = pipeline.run(file_path=str(path))
    return report


def test_the_statistics_cover_every_column_the_run_reported(tmp_path):
    report = _run(tmp_path)
    stats = agent_stats(report)

    assert set(stats.per_column) == set(report.entries)
    assert stats.columns == len(report.entries)
    assert stats.runtime_seconds == report.runtime_seconds


def test_attempts_and_outcomes_come_from_the_report(tmp_path):
    report = _run(tmp_path)
    stats = agent_stats(report)

    for name, entry in report.entries.items():
        column = stats.per_column[name]
        assert column.attempts == entry.attempts
        assert column.cleaned == entry.cleaned
        assert column.validated == entry.cleaning_validated


def test_tokens_are_counted_per_column_and_add_up_to_the_run(tmp_path):
    """Every token the run spent belongs to the column whose agents spent it."""
    report = _run(tmp_path)
    stats = agent_stats(report)

    assert report.per_task_usage, "the run must have attributed its tokens to columns"
    assert stats.total_tokens > 0
    assert stats.total_tokens == report.total_usage.total_tokens
    for name, column in stats.per_column.items():
        if not column.already_clean:
            assert column.total_tokens > 0, f"{name} cleaned without spending a token?"


def test_a_validator_rejection_is_counted(tmp_path):
    """A validator that sends the work back once leaves that in the statistics."""
    sent_back = {"done": False}

    def reject_the_first_validation(kind, messages):
        if kind == "validator" and not sent_back["done"]:
            sent_back["done"] = True
            return json.dumps(
                {
                    "needs_correction": True,
                    "feedback_target": "CODER",
                    "correction_instructions": "Try that again.",
                }
            )
        return None

    report = _run(tmp_path, script=reject_the_first_validation)
    stats = agent_stats(report)

    assert sent_back["done"], "the validator was never asked, so nothing was rejected"
    assert stats.validator_rejections == 1
    rejected = [name for name, column in stats.per_column.items() if column.validator_rejections]
    assert len(rejected) == 1
    assert stats.per_column[rejected[0]].attempts >= 2


def test_the_report_dictionary_does_not_grow_a_token_key(tmp_path):
    """Per-column tokens are kept beside the report, never inside the dictionary the GUI reads."""
    report = _run(tmp_path)
    data = report.to_dict()

    assert "per_task_usage" not in data
    for name in report.entries:
        assert "per_task_usage" not in data[name]
