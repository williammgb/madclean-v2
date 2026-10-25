"""After a rejection the system repairs the cells the validator named and asks it once more, instead of a
whole new Recommender → Coder → Validator round; a Coder that keeps failing and a validator that repeats
itself stop a column early."""

import asyncio
import json

import pandas as pd

from fake_llm import FakeLLMClient
from madclean.components.domain.schema import ColumnProfile
from madclean.components.multi_agent_cleaner.llm_coding import LLMCodingAgent
from madclean.components.multi_agent_cleaner.multi_agent_cleaning import MultiAgentCleaning
from madclean.config.loader import load_default_cleaning_config

CITY_CODE = (
    "import pandas as pd\n"
    "\n"
    "def clean_column(column):\n"
    "    return column.replace({'Saint Louis': 'St. Louis', 'Chicago IL': 'Chicago', 'chicago': 'Chicago'})\n"
)
# The code rewrites three spellings and misses the typo "Chicgo".
CITIES = ["Saint Louis"] * 20 + ["Chicago IL"] * 10 + ["Chicago"] * 55 + ["chicago"] * 10 + ["Chicgo"] * 5
BROKEN_CODE = "def clean_column(column):\n    raise ValueError('cannot read the column')\n"

SAINT_LOUIS = {"original": "Saint Louis", "cleaned": "St. Louis", "expected": "Saint Louis", "problem": "Abbreviated."}
CHICAGO_IL = {"original": "Chicago IL", "cleaned": "Chicago", "expected": "Chicago IL", "problem": "State dropped."}
TYPO = {"original": "Chicgo", "cleaned": "Chicgo", "expected": "Chicago", "problem": "Typo."}
APPROVAL = json.dumps({"needs_correction": False, "feedback_target": None, "correction_instructions": ""})


def rejection(issue_kind, cases, target="RECOMMENDER") -> str:
    answer = {"needs_correction": True, "feedback_target": target, "correction_instructions": "Fix it.",
              "cases": cases}
    if issue_kind is not None:
        answer["issue_kind"] = issue_kind
    return json.dumps(answer)


async def _execute_in_process(code_str, df, columns, timeout=30):
    """In place of the subprocess runner, which is slow to start; an error comes back as its message."""
    namespace: dict = {}
    try:
        exec(code_str, namespace)  # noqa: S102 - running the generated code is what is under test
        return namespace["clean_column"](df[columns].copy())
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def clean(monkeypatch, validator_answers, coder_code=CITY_CODE, **config_overrides):
    """Cleans one city column; returns (cleaned column, column report, the fake model, the dirty frame)."""
    monkeypatch.setattr(LLMCodingAgent, "_execute_code_async", staticmethod(_execute_in_process))
    answers = iter(validator_answers)
    fake = FakeLLMClient(coder_code=coder_code, script=lambda kind, _: next(answers) if kind == "validator" else None)
    config = load_default_cleaning_config()
    config.verbose = False
    config.sampling_seed = 7
    for name, value in config_overrides.items():
        setattr(config, name, value)
    cleaner = MultiAgentCleaning(llm_client=fake, llm_role="assistant", config=config)
    frame = pd.DataFrame({"city": CITIES})
    profile = ColumnProfile(name="city", semantic_type="NAMED_ENTITY", sample="Column 'city': ['Chicago']")

    _, cleaned, _ = asyncio.run(cleaner._run_column_cleaning_async(frame, "city", profile))
    return cleaned, cleaner.cleaning_report["city"], fake, frame


def step_ids(report) -> list[str]:
    return [step.id for step in report.trace_steps]


def step(report, step_id):
    return next(s for s in report.trace_steps if s.id == step_id)


def test_over_cleaning_is_repaired_with_one_extra_validator_call(monkeypatch):
    cleaned, report, fake, _ = clean(monkeypatch, [rejection("OVER_CLEANING", [SAINT_LOUIS, CHICAGO_IL]), APPROVAL])

    # The objected rows are back to their original; the rewrite nobody objected to stays.
    assert cleaned.tolist() == ["Saint Louis"] * 20 + ["Chicago IL"] * 10 + ["Chicago"] * 65 + ["Chicgo"] * 5
    assert len(fake.calls_for("recommender")) == 1 and fake.coder_calls == 1
    assert len(fake.calls_for("validator")) == 2
    assert step_ids(report) == ["recommender_1", "coder_1", "validator_1", "repair_1", "validator_repair_1"]
    repair = step(report, "repair_1")
    assert (repair.title, repair.status) == ("Repair (attempt 1)", "ok")
    assert repair.output == "Reverted 30 rows, fixed 0 rows from 2 cases; skipped 0."
    assert step(report, "validator_repair_1").title == "Validator (after repair)"
    assert report.cleaned is True and report.cleaning_validated is True and report.attempts == 1
    assert "_REPAIR_MAP" in report.generated_code
    second_prompt = fake.calls_for("validator")[1][-1]["content"]
    assert "### WHAT CHANGED SINCE YOUR LAST VERDICT" in second_prompt and "Reverted 30 rows" in second_prompt

    # A value the overview cut at 80 characters still matches; the repair key is the full value.
    long = "x" * 100
    table, reverted, _, skipped = MultiAgentCleaning._repair_table(
        pd.Series([long, "a"]), pd.Series(["y", "a"]),
        {"issue_kind": "OVER_CLEANING", "cases": [{"original": "x" * 80 + "…", "cleaned": "y"}]},
    )
    assert table == {long: long} and (reverted, skipped) == (1, 0)


def test_missed_errors_are_fixed_only_with_a_value_already_in_the_column(monkeypatch):
    cleaned, report, fake, _ = clean(monkeypatch, [rejection("MISSED_ERRORS", [TYPO]), APPROVAL])

    assert cleaned.tolist().count("Chicgo") == 0 and cleaned.tolist().count("Chicago") == 80
    assert step(report, "repair_1").output == "Reverted 0 rows, fixed 5 rows from 1 cases; skipped 0."
    assert len(fake.calls_for("recommender")) == 1 and report.cleaning_validated is True

    # "Chicago, Illinois" appears nowhere in the column, and an empty original is never repaired:
    # nothing applies, so a normal round follows.
    unknown = {**TYPO, "expected": "Chicago, Illinois"}
    empty = {"original": "<empty>", "cleaned": "<empty>", "expected": "Chicago", "problem": "Missing."}
    _, report, fake, _ = clean(monkeypatch, [rejection("MISSED_ERRORS", [unknown, empty]), APPROVAL])

    assert not any(step_id.startswith("repair") for step_id in step_ids(report))
    assert len(fake.calls_for("recommender")) == 2 and fake.coder_calls == 2
    assert report.attempts == 2 and report.cleaning_validated is True
    table, _, _, skipped = MultiAgentCleaning._repair_table(
        pd.Series(CITIES), pd.Series(CITIES), {"issue_kind": "MISSED_ERRORS", "cases": [unknown, empty]}
    )
    assert table == {} and skipped == 2


def test_a_code_bug_gets_no_repair(monkeypatch):
    _, report, fake, _ = clean(monkeypatch, [rejection("CODE_BUG", [SAINT_LOUIS], target="CODER"), APPROVAL])

    assert not any(step_id.startswith("repair") for step_id in step_ids(report))
    assert fake.coder_calls == 2 and len(fake.calls_for("validator")) == 2
    assert step_ids(report) == ["recommender_1", "coder_1", "validator_1", "coder_2", "validator_2"]


def test_a_coder_that_fails_two_rounds_in_a_row_stops_the_column(monkeypatch):
    cleaned, report, fake, _ = clean(monkeypatch, [], coder_code=BROKEN_CODE, max_coding_attempts=1)

    assert cleaned is None
    assert report.cleaned is False and report.attempts == 2
    assert report.reason == "Coder failed in 2 rounds in a row"
    assert len(fake.calls_for("recommender")) == 2 and fake.coder_calls == 2
    feedback = fake.calls_for("recommender")[1][-1]["content"]
    assert "The Coder's last error:" in feedback and "ValueError: cannot read the column" in feedback
    assert "ValueError: cannot read the column" in step(report, "coder_1").output


def test_the_same_objection_twice_finishes_with_what_was_not_objected_to(monkeypatch):
    answers = [
        rejection("OVER_CLEANING", [SAINT_LOUIS]),  # round 1: repaired …
        rejection("OVER_CLEANING", [CHICAGO_IL]),  # … and the repair rejected: a normal round follows
        rejection("OVER_CLEANING", [SAINT_LOUIS, CHICAGO_IL]),  # round 2: nothing new
    ]
    cleaned, report, fake, _ = clean(monkeypatch, answers)

    assert report.attempts == 2 and report.cleaned is True and report.cleaning_validated is False
    assert cleaned.tolist() == ["Saint Louis"] * 20 + ["Chicago IL"] * 10 + ["Chicago"] * 65 + ["Chicgo"] * 5
    assert len(fake.calls_for("validator")) == 3 and len(fake.calls_for("recommender")) == 2
    assert step_ids(report)[-2:] == ["coder_2", "validator_2"]


def test_the_repaired_code_run_on_the_dirty_column_gives_the_same_column(monkeypatch):
    # No issue kind: the rewrite is reverted and the missed error fixed, case by case.
    cleaned, report, _, frame = clean(monkeypatch, [rejection(None, [SAINT_LOUIS, TYPO]), APPROVAL])

    namespace: dict = {}
    exec(report.generated_code, namespace)  # noqa: S102 - the exported code is what is under test
    rerun = namespace["clean_column"](frame["city"].copy())

    assert step(report, "repair_1").output == "Reverted 20 rows, fixed 5 rows from 2 cases; skipped 0."
    assert rerun.tolist() == cleaned.tolist()
    assert cleaned.tolist() == ["Saint Louis"] * 20 + ["Chicago"] * 80
