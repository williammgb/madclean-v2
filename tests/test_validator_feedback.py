"""The validator sees whole columns, its feedback says what kind of problem it found, and a rejection on
the last attempt keeps the changes it did not object to."""

import asyncio
import json
import random
import re

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fake_llm import FakeLLMClient
from madclean.components.coordinator import prompts
from madclean.components.coordinator.prompt_generation import PromptGeneration
from madclean.components.domain.schema import ColumnProfile
from madclean.components.multi_agent_cleaner.llm_coding import LLMCodingAgent
from madclean.components.multi_agent_cleaner.llm_recommending import CodeOutputRecommendation
from madclean.components.multi_agent_cleaner.llm_validation import CodeOutputValidation
from madclean.components.multi_agent_cleaner.multi_agent_cleaning import MultiAgentCleaning
from madclean.config.loader import load_default_cleaning_config

OLD_SENTENCE = "errors from the original data remain"


def overview(dirty, cleaned, column_type="NAMED_ENTITY", **sizes) -> str:
    return PromptGeneration().create_prompt_validation(
        "city", pd.Series(dirty), pd.Series(cleaned), column_type, rng=random.Random(7), **sizes
    )


# --------------------------------------------------------------------------------------
# The column overview the validator reads
# --------------------------------------------------------------------------------------


def test_overview_counts_each_rewrite_once():
    dirty = ["Chicago"] * 90 + ["Chicago IL"] * 10
    cleaned = ["Chicago"] * 100

    prompt = overview(dirty, cleaned)

    assert "Chicago (90)" in prompt
    assert prompt.splitlines().count('"Chicago IL" → "Chicago" (10 rows)') == 1
    assert "10 of 100 cells changed (10.0%)" in prompt


def test_overview_without_changes_still_reaches_the_validator():
    prompt = overview(["a", "b", "b"], ["a", "b", "b"])

    assert "0 of 3 cells changed (0.0%)" in prompt
    assert "DISTINCT REWRITES" not in prompt
    assert "b (2)" in prompt


def test_overview_shows_missing_values_as_empty():
    prompt = overview(["N/A", "x", None], [None, "x", "y"])

    assert '"N/A" → <empty> (1 rows)' in prompt
    assert '<empty> → "y" (1 rows)' in prompt
    assert "1 cells emptied" in prompt


def test_overview_cuts_long_rewrite_lists_and_long_values():
    dirty = [f"{i} kg" for i in range(200)] + ["x" * 100]
    cleaned = [str(i) for i in range(200)] + ["x" * 100]

    prompt = overview(dirty, cleaned, column_type="DIRTY_INTEGER", max_sample_size=150)

    assert "200 of 201 cells changed (99.5%)" in prompt
    assert sum(" rows)" in line for line in prompt.splitlines()) == 150
    assert "… and 50 more distinct rewrites covering 50 rows" in prompt
    assert '"' + "x" * 80 + '…"' in prompt
    assert "x" * 81 not in prompt


cell = st.one_of(st.none(), st.sampled_from(["a", "b", "a b", "A", "1", "1.0", ""]))


@settings(max_examples=150, deadline=None)
@given(pairs=st.lists(st.tuples(cell, cell), max_size=60), cap=st.integers(0, 8))
def test_overview_accounts_for_every_changed_cell(pairs, cap):
    dirty = [d for d, _ in pairs]
    cleaned = [c for _, c in pairs]
    changed = sum(PromptGeneration._cell_changed(d, c) for d, c in pairs)
    emptied = sum(d is not None and c is None for d, c in pairs)

    prompt = overview(dirty, cleaned, max_sample_size=cap)

    rows = sum(int(n) for n in re.findall(r" \((\d+) rows\)$", prompt, flags=re.M))
    rows += sum(int(n) for n in re.findall(r"covering (\d+) rows$", prompt, flags=re.M))
    assert rows == changed
    assert f"{changed} of {len(pairs)} cells changed" in prompt
    assert f"{emptied} cells emptied" in prompt


# --------------------------------------------------------------------------------------
# Prompts and answer shapes
# --------------------------------------------------------------------------------------


def test_every_recommender_prompt_has_the_core_rule():
    for name, template in prompts.RECOMMENDATION_PROMPT_TEMPLATES.items():
        assert "When unsure, leave the value unchanged." in template, name
        assert "set it to NaN" not in template.lower(), name


def test_every_validator_prompt_asks_for_the_issue_kind_and_cases():
    for name, template in prompts.VALIDATION_PROMPT_TEMPLATES.items():
        if not template:
            continue
        for word in ('"issue_kind"', '"cases"', "OVER_CLEANING", "MISSED_ERRORS", "FORMAT_CHANGE", "CODE_BUG"):
            assert word in template, (name, word)


def test_analysis_is_first_field():
    assert list(CodeOutputRecommendation.model_fields)[0] == "analysis"
    assert list(CodeOutputValidation.model_fields)[0] == "analysis"


def test_old_validator_answer_still_parses():
    answer = CodeOutputValidation(needs_correction=True, feedback_target="RECOMMENDER", correction_instructions="x")

    assert (answer.analysis, answer.issue_kind, answer.cases) == ("", None, None)


def test_numeric_case_values_parse_as_text():
    answer = CodeOutputValidation(
        needs_correction=True, feedback_target="RECOMMENDER", correction_instructions="x",
        cases=[{"original": "12 kg", "cleaned": 12, "expected": 12.5, "problem": "rounded"}],
    )

    assert answer.cases[0].cleaned == "12" and answer.cases[0].expected == "12.5"


# --------------------------------------------------------------------------------------
# The cleaning loop, with a scripted model
# --------------------------------------------------------------------------------------

CITY_CODE = (
    "import pandas as pd\n"
    "\n"
    "def clean_column(column):\n"
    "    return column.replace({'Saint Louis': 'St. Louis', 'Chicago IL': 'Chicago'})\n"
)
CITIES = ["Saint Louis"] * 20 + ["Chicago IL"] * 10 + ["Chicago"] * 70


def rejection(issue_kind=None, cases=None, target="RECOMMENDER") -> str:
    answer = {"needs_correction": True, "feedback_target": target, "correction_instructions": "Fix it."}
    if issue_kind is not None:
        answer["issue_kind"] = issue_kind
    if cases is not None:
        answer["cases"] = cases
    return json.dumps(answer)


APPROVAL = json.dumps({"needs_correction": False, "feedback_target": None, "correction_instructions": ""})
SAINT_LOUIS = {"original": "Saint Louis", "cleaned": "St. Louis", "expected": "Saint Louis",
               "problem": "Abbreviated from outside knowledge."}


async def _execute_in_process(code_str, df, columns, timeout=30):
    """In place of the subprocess runner, which is slow to start."""
    namespace: dict = {}
    exec(code_str, namespace)
    return namespace["clean_column"](df[columns].copy())


def clean_city(monkeypatch, validator_answers, **config_overrides):
    """Cleans one city column; returns (cleaned column, column report, the fake model)."""
    monkeypatch.setattr(LLMCodingAgent, "_execute_code_async", staticmethod(_execute_in_process))
    answers = iter(validator_answers)
    fake = FakeLLMClient(coder_code=CITY_CODE, script=lambda kind, _: next(answers) if kind == "validator" else None)
    config = load_default_cleaning_config()
    config.verbose = False
    config.sampling_seed = 7
    for name, value in config_overrides.items():
        setattr(config, name, value)
    cleaner = MultiAgentCleaning(llm_client=fake, llm_role="assistant", config=config)
    frame = pd.DataFrame({"city": CITIES})
    profile = ColumnProfile(name="city", semantic_type="NAMED_ENTITY", sample="Column 'city': ['Chicago']")

    _, cleaned, _ = asyncio.run(cleaner._run_column_cleaning_async(frame, "city", profile))
    return cleaned, cleaner.cleaning_report["city"], fake


def feedback_to_recommender(fake: FakeLLMClient) -> str:
    second_call = fake.calls_for("recommender")[1]
    return second_call[-1]["content"]


def test_over_cleaning_feedback_does_not_say_errors_remain(monkeypatch):
    _, report, fake = clean_city(monkeypatch, [rejection("OVER_CLEANING", [SAINT_LOUIS]), APPROVAL])

    message = feedback_to_recommender(fake)
    assert OLD_SENTENCE not in message
    assert "changed values that were already correct" in message
    assert "Saint Louis → St. Louis → Saint Louis: Abbreviated from outside knowledge." in message
    assert report.cleaning_validated is True


def test_missed_errors_feedback_keeps_the_old_sentence(monkeypatch):
    _, _, fake = clean_city(monkeypatch, [rejection("MISSED_ERRORS"), APPROVAL])

    assert OLD_SENTENCE in feedback_to_recommender(fake)


def test_feedback_without_issue_kind_is_neutral(monkeypatch):
    _, _, fake = clean_city(monkeypatch, [rejection(), APPROVAL])

    message = feedback_to_recommender(fake)
    assert "the reviewer found problems" in message
    assert OLD_SENTENCE not in message


def test_code_bug_feedback_goes_to_the_coder(monkeypatch):
    _, _, fake = clean_city(monkeypatch, [rejection("CODE_BUG", target="CODER"), APPROVAL])

    assert len(fake.calls_for("recommender")) == 1
    assert "The previous code was almost correct" in fake.calls_for("coder")[1][-1]["content"]


def test_last_attempt_reverts_only_objected_rows(monkeypatch):
    cleaned, report, _ = clean_city(
        monkeypatch, [rejection("OVER_CLEANING", [SAINT_LOUIS])], max_cleaning_attempts=1
    )

    assert cleaned.tolist() == ["Saint Louis"] * 20 + ["Chicago"] * 80
    assert report.cleaned is True and report.cleaning_validated is False
    assert report.trace_steps[-1].status == "needs_correction"


def test_last_attempt_keeps_everything_for_missed_errors(monkeypatch):
    cleaned, report, _ = clean_city(monkeypatch, [rejection("MISSED_ERRORS")], max_cleaning_attempts=1)

    assert cleaned.tolist() == ["St. Louis"] * 20 + ["Chicago"] * 80
    assert report.cleaned is True and report.cleaning_validated is False


@pytest.mark.parametrize("answer", [
    rejection("OVER_CLEANING", [{"original": "Paris", "cleaned": "Lyon", "expected": "Paris", "problem": "?"}]),
    rejection(),
])
def test_last_attempt_without_matching_cases_leaves_the_column_uncleaned(monkeypatch, answer):
    cleaned, report, _ = clean_city(monkeypatch, [answer], max_cleaning_attempts=1)

    assert cleaned is None
    assert report.cleaned is False
