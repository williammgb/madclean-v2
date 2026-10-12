"""Dependency fixes come back as a table the system turns into code and runs, with no Coder in between."""

import asyncio
import json
import re

import pandas as pd
import pytest

from fake_llm import FakeLLMClient
from madclean.components.coordinator.prompt_generation import PromptGeneration
from madclean.components.dataprofiler.functional_dependencies import FunctionalDependencies
from madclean.components.domain.schema import FDResult, MultiColumnTask
from madclean.components.multi_agent_cleaner.code_checks import dependency_code_from_table, dependency_table
from madclean.components.multi_agent_cleaner.llm_coding import LLMCodingAgent
from madclean.components.multi_agent_cleaner.multi_agent_cleaning import MultiAgentCleaning
from madclean.config.loader import load_default_cleaning_config

FD_KEY = "brewery_id → city"


async def _execute_in_process(code_str, df, columns, timeout=30):
    """The subprocess runner without its second of start-up: same input copy, same type check."""
    data = df[list(columns)].copy()
    namespace: dict = {}
    try:
        exec(code_str, namespace)  # noqa: S102 - running generated code is what is under test
        result = namespace["clean_column"](data)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"clean_column must return a pandas DataFrame, but returned {type(result).__name__}.")
        return result
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


@pytest.fixture(autouse=True)
def in_process(monkeypatch):
    monkeypatch.setattr(LLMCodingAgent, "_execute_code_async", staticmethod(_execute_in_process))


def breweries() -> pd.DataFrame:
    """Brewery 80 is in Portland five times and "Portland ME" once; brewery 81 is consistent."""
    return pd.DataFrame({
        "brewery_id": [80, 80, 80, 80, 80, 80, 81, 81, 80],
        "city": ["Portland", "Portland", "Portland", "Portland ME", "Portland", "Portland", "Bend", "Bend", None],
    })


def fd_task(df: pd.DataFrame) -> MultiColumnTask:
    task = MultiColumnTask(
        task_type="FD", target_columns=["brewery_id", "city"], verbose_key=FD_KEY,
        data=FDResult(lhs="brewery_id", rhs="city", score=0.95),
    )
    return FunctionalDependencies().get_data(df, task)


def fd_answer(corrections=None, impute_missing=False, skipped=None) -> str:
    return json.dumps({
        "analysis": "Brewery 80 is in Portland; one row adds a state code.",
        "summary": "Each brewery has one city.",
        "corrections": corrections,
        "skipped_lhs_values": skipped,
        "impute_missing": impute_missing,
    })


def run_fd(df: pd.DataFrame, answers: list[str], validator=None, **config_overrides):
    config = load_default_cleaning_config()
    config.verbose = False
    for name, value in config_overrides.items():
        setattr(config, name, value)
    counts = {"fd_recommender": 0, "validator": 0}

    def script(kind, messages):
        if kind == "fd_recommender":
            counts[kind] += 1
            return answers[min(counts[kind] - 1, len(answers) - 1)]
        if kind == "validator" and validator is not None:
            counts[kind] += 1
            return validator[min(counts[kind] - 1, len(validator) - 1)]
        return None

    fake = FakeLLMClient(script=script)
    loop = MultiAgentCleaning(llm_client=fake, llm_role="assistant", config=config)
    _, cleaned, _ = asyncio.run(loop._run_multi_col_cleaning_async(df, fd_task(df)))
    return cleaned, loop.cleaning_report[FD_KEY], fake


def test_a_correction_changes_exactly_the_conflicting_row():
    df = breweries()

    cleaned, report, fake = run_fd(df, [fd_answer([{"lhs_value": "80", "correct_rhs": "Portland"}])])

    changed = [i for i in df.index if str(df.loc[i, "city"]) != str(cleaned.loc[i, "city"])]
    assert changed == [3]
    assert cleaned.loc[3, "city"] == "Portland"
    assert pd.isna(cleaned.loc[8, "city"])
    assert fake.coder_calls == 0
    assert report.cleaned is True
    assert "corrections = {" in report.generated_code


def test_impute_missing_fills_an_empty_city_from_the_breweries_one_city():
    df = breweries()

    cleaned, report, fake = run_fd(df, [fd_answer([{"lhs_value": 80, "correct_rhs": "Portland"}], impute_missing=True)])

    assert cleaned.loc[8, "city"] == "Portland"
    assert cleaned.loc[3, "city"] == "Portland"
    assert cleaned["city"].tolist()[6:8] == ["Bend", "Bend"]
    assert fake.coder_calls == 0


def test_impute_missing_leaves_a_cell_whose_left_value_still_has_two_cities():
    df = breweries()

    cleaned, _, _ = run_fd(df, [fd_answer(None, impute_missing=True)])

    # Brewery 80 still has "Portland" and "Portland ME", so its empty city stays empty.
    assert pd.isna(cleaned.loc[8, "city"])
    assert cleaned.loc[3, "city"] == "Portland ME"


def test_no_corrections_and_no_imputation_finishes_as_nothing_to_change():
    df = breweries()

    cleaned, report, fake = run_fd(df, [fd_answer(None, impute_missing=False)])

    assert cleaned is None
    assert report.cleaned is False and report.reason is None and not report.generated_code
    assert report.trace_steps[-1].output.startswith("No corrections and no imputation")
    assert fake.coder_calls == 0


def test_a_left_value_that_matches_no_row_is_named_in_the_trace():
    df = breweries()

    cleaned, report, _ = run_fd(df, [fd_answer([
        {"lhs_value": "80.0", "correct_rhs": "Portland"},
        {"lhs_value": "80", "correct_rhs": "Portland"},
    ])])

    step = report.trace_steps[-1]
    assert step.id == "fd_code_1"
    assert 'Unmatched left-hand values (no row has them): "80.0"' in step.output
    assert cleaned.loc[3, "city"] == "Portland"


def test_feedback_meant_for_the_coder_goes_to_the_recommender():
    df = breweries()
    validator = [
        json.dumps({"needs_correction": True, "feedback_target": "CODER", "correction_instructions": "Fill the empty city."}),
        json.dumps({"needs_correction": False, "feedback_target": None, "correction_instructions": ""}),
    ]

    cleaned, report, fake = run_fd(
        df,
        [fd_answer([{"lhs_value": "80", "correct_rhs": "Portland"}]),
         fd_answer([{"lhs_value": "80", "correct_rhs": "Portland"}], impute_missing=True)],
        validator=validator,
        enable_validation_multi=True,
    )

    assert fake.coder_calls == 0
    recommender_calls = fake.calls_for("fd_recommender")
    assert len(recommender_calls) == 2
    assert "Fill the empty city." in recommender_calls[1][-1]["content"]
    assert report.cleaning_validated is True and report.attempts == 2
    assert cleaned.loc[8, "city"] == "Portland"


def test_a_numeric_right_hand_column_gets_numbers():
    df = pd.DataFrame({"zip": ["97201", "97201", "97201", "97702"], "area": [503, 503, 504, 541]})
    table, unusable = dependency_table(
        [{"lhs_value": "97201", "correct_rhs": "503"}, {"lhs_value": "97702", "correct_rhs": "five"}],
        df["area"].dtype,
    )
    namespace: dict = {}
    exec(dependency_code_from_table("zip", "area", table, impute_missing=False), namespace)  # noqa: S102

    cleaned = namespace["clean_column"](df)

    assert table == {"97201": 503} and unusable == ["97702"]
    assert cleaned["area"].tolist() == [503, 503, 503, 541]
    assert pd.api.types.is_integer_dtype(cleaned["area"])


def test_violations_are_sorted_by_rows_and_carry_their_row_total():
    df = pd.DataFrame({
        "brewery_id": [1, 1, 2, 2, 2, 2, 3, 3, 3],
        "city": ["A", "B", "C", "C", "C", "D", "E", "E", "F"],
    })

    violations = fd_task(df).data.violation_data["violations"]

    assert [(v["lhs"], v["rows"]) for v in violations] == [(2, 4), (3, 3), (1, 2)]


def test_the_prompt_lists_the_first_100_violations_and_counts_the_rest():
    violations = [
        {"lhs": f"b{i}", "rhs_conflicts": [("X", 2), ("Y", 1)], "rows": 3, "context": [[f"b{i}", "X"], [f"b{i}", "Y"]]}
        for i in range(150)
    ]
    df = pd.DataFrame({"brewery_id": ["b0", "b0"], "city": ["X", "Y"]})
    fd = FDResult(
        lhs="brewery_id", rhs="city", score=0.95, violations_count=150,
        violation_data={"count": 150, "violations": violations}, imputation_data={"count": 0},
    )
    task = MultiColumnTask(task_type="FD", target_columns=["brewery_id", "city"], verbose_key=FD_KEY, data=fd)

    prompt = PromptGeneration().create_prompt_recommender_multi_col(df, task, max_violations=100)

    assert len(re.findall(r"^\{'lhs': 'b\d+'", prompt, flags=re.MULTILINE)) == 100
    assert "{'lhs': 'b99'" in prompt and "{'lhs': 'b100'" not in prompt
    assert "50 more violations covering 150 rows are not shown and will be left unchanged." in prompt
    assert '"corrections"' in prompt and '"impute_missing"' in prompt
