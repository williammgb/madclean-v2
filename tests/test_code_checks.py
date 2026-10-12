"""The checks around the Coder: its examples, its crashes, cells it empties, and the exact value table."""

import asyncio
import json

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fake_llm import FakeLLMClient
from madclean.components.domain.schema import ColumnProfile
from madclean.components.multi_agent_cleaner.code_checks import (
    emptied_share,
    example_cases,
    keep_whole_numbers,
    parse_example,
    values_match,
    wrap_with_value_map,
)
from madclean.components.multi_agent_cleaner.llm_coding import LLMCodingAgent
from madclean.components.multi_agent_cleaner.multi_agent_cleaning import MultiAgentCleaning
from madclean.config.loader import load_default_cleaning_config

UPPER_CODE = "import pandas as pd\n\ndef clean_column(column):\n    return column.str.upper()\n"
TITLE_CODE = "import pandas as pd\n\ndef clean_column(column):\n    return column.str.title()\n"
IDENTITY_CODE = "def clean_column(column):\n    return column\n"
CRASH_CODE = (
    "def clean_column(column):\n"
    "    values = column.copy()\n"
    "    raise ValueError(f'cannot read {values.iloc[0]!r}')\n"
)


def code_answer(code: str) -> str:
    return f"```python\n{code}```"


async def _execute_in_process(code_str, df, columns, timeout=30):
    """The subprocess runner without its second of start-up: same input copy, same type check."""
    data = df[[columns]].copy()[columns]
    namespace: dict = {}
    try:
        exec(code_str, namespace)  # noqa: S102 - running generated code is what is under test
        result = namespace["clean_column"](data)
        if not isinstance(result, pd.Series):
            raise TypeError(f"clean_column must return a pandas Series, but returned {type(result).__name__}.")
        return result
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


@pytest.fixture
def in_process(monkeypatch):
    """Keeps the fast gate fast; the crash tests leave it out and run the real subprocess."""
    monkeypatch.setattr(LLMCodingAgent, "_execute_code_async", staticmethod(_execute_in_process))


def _config(**overrides):
    config = load_default_cleaning_config()
    config.verbose = False
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


async def _no_tokens(input_tokens, output_tokens):
    return None


def _coder(answers: list[str], **config):
    """A coding agent whose model gives these answers in order, the last one repeating."""
    seen = []

    def script(kind, messages):
        seen.append(messages)
        return answers[min(len(seen) - 1, len(answers) - 1)]

    fake = FakeLLMClient(script=script)
    return LLMCodingAgent(fake, "assistant", _no_tokens, _config(**config)), fake


def _recommendation(**fields) -> dict:
    data = {
        "summary": "City names.",
        "error_types": ["casing"],
        "examples_clean": [],
        "examples_dirty": [],
        "cleaning_instructions": ["Capitalise each word."],
        "value_mapping": None,
    }
    data.update(fields)
    return data


def _fix_ups(fake: FakeLLMClient) -> list[str]:
    """The messages the Coder was sent after its first answer: everything the checks fed back."""
    last_call = fake.calls_for("coder")[-1]
    return [m["content"] for m in last_call[2:] if m["role"] == "user"]


def _clean(agent, df, col, data):
    return asyncio.run(agent.clean_column_async(df, col, "NAMED_ENTITY", recommender_data=data))


# ---------------------------------------------------------------------------------------------
# Example check
# ---------------------------------------------------------------------------------------------


def test_code_that_breaks_an_example_gets_the_case_back_once(in_process):
    df = pd.DataFrame({"city": ["chicago", "Boston", "denver", "Austin"]})
    agent, fake = _coder([code_answer(UPPER_CODE), code_answer(TITLE_CODE)])
    data = _recommendation(examples_dirty=["chicago → Chicago"], examples_clean=["Boston"])

    cleaned, code, _, note = _clean(agent, df, "city", data)

    fix_ups = _fix_ups(fake)
    assert [m for m in fix_ups if "chicago → Chicago → CHICAGO" in m] == fix_ups and len(fix_ups) == 1
    assert "Boston → Boston → BOSTON" in fix_ups[0]
    assert cleaned.tolist() == ["Chicago", "Boston", "Denver", "Austin"]
    assert note == "Example check: 2 of 2 passed"
    assert code == TITLE_CODE.strip()


def test_after_the_last_attempt_the_code_closest_to_the_examples_goes_on(in_process):
    df = pd.DataFrame({"city": ["chicago", "Boston"]})
    agent, fake = _coder([code_answer(UPPER_CODE), code_answer(IDENTITY_CODE)], max_coding_attempts=2)
    data = _recommendation(examples_dirty=["chicago → Chicago"], examples_clean=["Boston"])

    cleaned, code, _, note = _clean(agent, df, "city", data)

    # Upper-casing fails both examples, leaving the values alone fails one: the identity goes on.
    assert code == IDENTITY_CODE.strip()
    assert cleaned.tolist() == ["chicago", "Boston"]
    assert note == "Example check: 1 of 2 passed"


def test_examples_the_value_table_covers_are_not_held_against_the_code(in_process):
    df = pd.DataFrame({"city": ["chicago", "Chicago IL", "Boston"]})
    agent, fake = _coder([code_answer(TITLE_CODE)])
    # The example says "Chicago IL" stays; the table says otherwise and wins.
    data = _recommendation(
        examples_dirty=["chicago → Chicago"],
        examples_clean=["Chicago IL", "Boston"],
        value_mapping=[{"from_value": "Chicago IL", "to_value": "Chicago"}],
    )

    cleaned, _, _, note = _clean(agent, df, "city", data)

    assert fake.coder_calls == 1
    assert cleaned.tolist() == ["Chicago", "Chicago", "Boston"]
    assert note == "Example check: 2 of 2 passed"


def test_no_usable_examples_means_no_check(in_process):
    df = pd.DataFrame({"city": ["chicago"]})
    agent, fake = _coder([code_answer(UPPER_CODE)])
    data = _recommendation(examples_dirty=["capitalise every word", "None → <empty>"])

    cleaned, _, _, note = _clean(agent, df, "city", data)

    assert cleaned.tolist() == ["CHICAGO"] and note is None and fake.coder_calls == 1


def test_parse_example_splits_on_the_first_arrow_and_reads_missing_words():
    assert parse_example('"chicago" → "Chicago"') == ("chicago", "Chicago")
    assert parse_example("a → b → c") == ("a", "b → c")
    assert parse_example("x -> y") == ("x", "y")
    assert parse_example("N/A → NaN") == ("N/A", None)
    assert parse_example("'' → <empty>") is None
    assert parse_example("NULL → 0") is None
    assert parse_example("no arrow here") is None


def test_examples_compare_numbers_as_numbers_and_text_trimmed():
    assert values_match(12, "12")
    assert values_match(12.0000000001, "12")
    assert values_match(" Chicago ", "Chicago")
    assert values_match(float("nan"), None)
    assert values_match("", None)
    assert not values_match("CHICAGO", "Chicago")
    assert not values_match(None, "Chicago")


def test_example_cases_hold_clean_values_unchanged_and_skip_missing_inputs():
    data = _recommendation(examples_dirty=["NaN → x", "a → b"], examples_clean=["c", "None"])

    assert example_cases(data) == [("a", "b"), ("c", "c")]


def test_example_inputs_take_the_columns_number_type_when_they_all_convert(in_process):
    df = pd.DataFrame({"abv": [5, 6, 7]})
    code = "def clean_column(column):\n    assert column.dtype.kind == 'i'\n    return column + 1\n"
    agent, _ = _coder([code_answer(code)])
    data = _recommendation(examples_dirty=["5 → 6"], examples_clean=[])

    cleaned, _, _, note = asyncio.run(agent.clean_column_async(df, "abv", "INTEGER", recommender_data=data))

    assert cleaned.tolist() == [6, 7, 8] and note == "Example check: 1 of 1 passed"


# ---------------------------------------------------------------------------------------------
# Crash messages
# ---------------------------------------------------------------------------------------------


def test_a_crash_reaches_the_coder_with_its_line_and_value():
    df = pd.DataFrame({"city": ["chicago", "Boston"]})
    agent, fake = _coder([code_answer(CRASH_CODE), code_answer(TITLE_CODE)])

    _clean(agent, df, "city", _recommendation())

    (fix_up,) = _fix_ups(fake)
    assert "line 3" in fix_up
    assert "ValueError: cannot read 'chicago'" in fix_up


def test_a_crash_message_is_the_last_fifteen_lines_of_the_error_output():
    noisy = (
        "import sys\n\n"
        "def clean_column(column):\n"
        "    for i in range(40):\n"
        "        print(f'noise {i}', file=sys.stderr)\n"
        "    return 1 / 0\n"
    )

    message = LLMCodingAgent._execute_code(noisy, pd.DataFrame({"a": [1]}), "a")

    lines = message.splitlines()
    assert len(lines) == 15
    assert lines[-1] == "ZeroDivisionError: division by zero"
    assert "noise 0" not in message


# ---------------------------------------------------------------------------------------------
# Emptied-cells guard
# ---------------------------------------------------------------------------------------------

BLANK_60_CODE = (
    "import pandas as pd\n\n"
    "def clean_column(column):\n"
    "    result = column.copy()\n"
    "    result.iloc[:60] = None\n"
    "    return result\n"
)


def test_code_that_empties_most_filled_cells_is_rejected(in_process):
    df = pd.DataFrame({"code": [f"value {i}" for i in range(100)]})
    agent, fake = _coder([code_answer(BLANK_60_CODE), code_answer(IDENTITY_CODE)])

    cleaned, _, _, _ = _clean(agent, df, "code", _recommendation())

    (fix_up,) = _fix_ups(fake)
    assert "emptied 60 of 100 filled cells (60.0%)" in fix_up
    assert "Only missing-value placeholders may become empty." in fix_up
    assert cleaned.notna().all()


def test_removing_a_unit_from_every_value_passes_the_guard(in_process):
    df = pd.DataFrame({"ounces": ["12 oz"] * 40 + ["16 oz"] * 10})
    code = "import pandas as pd\n\ndef clean_column(column):\n    return column.str.replace(' oz', '').astype(int)\n"
    agent, fake = _coder([code_answer(code)])

    cleaned, _, _, _ = _clean(agent, df, "ounces", _recommendation(examples_dirty=["12 oz → 12"]))

    assert fake.coder_calls == 1 and cleaned.tolist() == [12] * 40 + [16] * 10


def test_blanking_a_column_that_is_mostly_placeholders_passes(in_process):
    df = pd.DataFrame({"state": ["N/A"] * 70 + ["OR"] * 30})
    code = "def clean_column(column):\n    return column.where(column != 'N/A')\n"
    agent, fake = _coder([code_answer(code)])

    cleaned, _, _, _ = _clean(agent, df, "state", _recommendation())

    assert fake.coder_calls == 1 and int(cleaned.isna().sum()) == 70


def test_blanking_a_column_that_holds_only_the_word_empty_passes(in_process):
    # hospital's Address2 and Sample hold the word "empty" in every row; the truth is a blank cell.
    df = pd.DataFrame({"Address2": ["empty"] * 100})
    code = "def clean_column(column):\n    return column.where(column != 'empty')\n"
    agent, fake = _coder([code_answer(code)])

    cleaned, _, _, _ = _clean(agent, df, "Address2", _recommendation())

    assert fake.coder_calls == 1 and int(cleaned.isna().sum()) == 100


def test_whole_numbers_turned_into_decimals_go_back_to_whole_numbers():
    # rayyan's article_jvolumn: "64" came out as 64.0 once one cell became empty.
    original = pd.Series(["64", "12", "abc", "", "7"], dtype=object)
    cleaned = pd.Series([64.0, 12.0, None, None, 7.0])

    kept = keep_whole_numbers(original, cleaned)

    assert str(kept.dtype) == "Int64"
    assert kept.astype(str).tolist() == ["64", "12", "<NA>", "<NA>", "7"]


@pytest.mark.parametrize(
    "original, cleaned",
    [
        (["7.0", "8.0", "6.5"], [7.0, 8.0, 7.0]),  # the original numbers were written as decimals
        (["1.5", "2", "3"], [1.5, 2.0, 3.0]),  # a real fraction stays
        (["x", "y", "z"], [1.0, 2.0, 3.0]),  # no original number to judge by
        (["1", "2", "3"], ["1", "2", "3"]),  # not a decimal column
        ([4.5, None, 30.0], [5.0, None, 30.0]),  # loaded as real decimals
    ],
)
def test_decimal_columns_stay_decimal(original, cleaned):
    cleaned = pd.Series(cleaned)
    original = pd.Series(original, dtype=None if isinstance(original[0], float) else object)

    assert keep_whole_numbers(original, cleaned) is cleaned


def test_a_whole_number_column_loaded_as_decimals_because_of_blanks_is_whole_again():
    # rayyan's article_jvolumn loads as 64.0 because some cells are blank.
    original = pd.Series([64.0, None, 12.0])

    assert keep_whole_numbers(original, original.copy()).astype(str).tolist() == ["64", "<NA>", "12"]


def test_inputs_the_recommender_maps_to_empty_do_not_count():
    original = pd.Series(["gone"] * 8 + ["kept"] * 2)
    cleaned = pd.Series([None] * 8 + ["kept"] * 2)

    assert emptied_share(original, cleaned) == (8, 10)
    assert emptied_share(original, cleaned, ["gone"]) == (0, 2)


@settings(max_examples=100, deadline=None)
@given(
    values=st.lists(st.one_of(st.none(), st.text(max_size=6), st.sampled_from(["N/A", " tbd ", "x", "-"])), max_size=30),
    blank=st.lists(st.booleans(), max_size=30),
)
def test_the_guard_counts_only_cells_that_become_empty(values, blank):
    original = pd.Series(values, dtype=object)
    blanked = [None if i < len(blank) and blank[i] else v for i, v in enumerate(values)]
    rewritten = [None if v is None else f"<{v}>" for v in values]

    emptied, filled = emptied_share(original, pd.Series(blanked, dtype=object))

    assert 0 <= emptied <= filled <= len(values)
    # Rewriting every value without emptying any never counts, however many cells change.
    assert emptied_share(original, pd.Series(rewritten, dtype=object))[0] == 0


# ---------------------------------------------------------------------------------------------
# Value table
# ---------------------------------------------------------------------------------------------


def _run_in_process(code: str, column: pd.Series) -> pd.Series:
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102 - the generated code is what is under test
    return namespace["clean_column"](column)


def test_a_value_table_alone_changes_only_its_values_and_calls_no_coder(in_process):
    frame = pd.DataFrame({"city": ["Chicago", "Chicago IL", "Boston", "Chicago IL", "chicago il"]})
    answer = json.dumps({
        "analysis": "Two rows carry a state code the rest do not.",
        "is_clean": False,
        "summary": "City names.",
        "error_types": ["state code"],
        "examples_clean": ["Chicago"],
        "examples_dirty": ["Chicago IL → Chicago"],
        "cleaning_instructions": None,
        "value_mapping": [{"from_value": "Chicago IL", "to_value": "Chicago"}],
    })
    fake = FakeLLMClient(script=lambda kind, messages: answer if kind == "recommender" else None)
    loop = MultiAgentCleaning(llm_client=fake, llm_role="assistant", config=_config())
    profile = ColumnProfile(name="city", semantic_type="CATEGORICAL", sample="Column 'city'")

    _, cleaned, _ = asyncio.run(loop._run_column_cleaning_async(frame, "city", profile))

    assert cleaned.tolist() == ["Chicago", "Chicago", "Boston", "Chicago", "chicago il"]
    assert fake.coder_calls == 0
    report = loop.cleaning_report["city"]
    assert "_VALUE_MAP" in report.generated_code
    assert report.cleaned is True
    coder_step = next(step for step in report.trace_steps if step.id == "coder_1")
    assert coder_step.output.startswith("Value table only: the Coder was not called.")


def test_the_coder_prompt_lists_the_table_and_says_not_to_implement_it(in_process):
    df = pd.DataFrame({"city": ["chicago", "Chicago IL"]})
    agent, fake = _coder([code_answer(TITLE_CODE)])
    data = _recommendation(value_mapping=[{"from_value": "Chicago IL", "to_value": None}])

    _clean(agent, df, "city", data)

    prompt = fake.calls_for("coder")[0][1]["content"]
    assert 'EXACT REPLACEMENTS APPLIED AFTER YOUR FUNCTION' in prompt
    assert '- "Chicago IL" → <empty>' in prompt
    assert "Do NOT implement them in your code" in prompt


def test_the_coder_prompt_says_none_without_a_table(in_process):
    df = pd.DataFrame({"city": ["chicago"]})
    agent, fake = _coder([code_answer(TITLE_CODE)])

    _clean(agent, df, "city", _recommendation())

    prompt = fake.calls_for("coder")[0][1]["content"]
    assert "EXACT REPLACEMENTS APPLIED AFTER YOUR FUNCTION" in prompt
    assert prompt.split("Do NOT implement them in your code:\n")[1].startswith("(none)")


def test_the_table_is_applied_on_the_original_values_after_the_code():
    code = wrap_with_value_map(UPPER_CODE, [
        {"from_value": "chicago il", "to_value": "Chicago"},
        {"from_value": "nowhere", "to_value": "Somewhere"},
        {"from_value": "unknown", "to_value": None},
    ])

    result = _run_in_process(code, pd.Series(["chicago il", "boston", "unknown"]))

    # The code upper-cases everything; the table then sets its rows from their original value.
    assert result.tolist()[:2] == ["Chicago", "BOSTON"] and pd.isna(result.iloc[2])
    assert "_VALUE_MAP" in code and "def _clean_by_code(" in code


def test_a_numeric_column_matches_the_table_on_its_text_form():
    code = wrap_with_value_map(None, [{"from_value": "12", "to_value": "13"}, {"from_value": "7", "to_value": None}])

    floats = _run_in_process(code, pd.Series([12.0, 7.0, 3.0]))
    ints = _run_in_process(code, pd.Series([12, 7, 3]))

    # "12.0" is not "12": the float column is untouched, the integer column is mapped and stays numeric.
    assert floats.tolist() == [12.0, 7.0, 3.0]
    assert ints.iloc[0] == 13 and pd.isna(ints.iloc[1]) and ints.iloc[2] == 3
    assert pd.api.types.is_numeric_dtype(ints)


def test_a_table_entry_that_matches_nothing_changes_nothing():
    column = pd.Series(["a", "b", None])

    result = _run_in_process(wrap_with_value_map(None, [{"from_value": "z", "to_value": "y"}]), column)

    pd.testing.assert_series_equal(result, column)


def test_only_the_first_500_table_entries_are_used():
    mapping = [{"from_value": f"v{i}", "to_value": "x"} for i in range(600)]

    result = _run_in_process(wrap_with_value_map(None, mapping), pd.Series(["v499", "v500"]))

    assert result.tolist() == ["x", "v500"]


@settings(max_examples=60, deadline=None)
@given(
    values=st.lists(st.one_of(st.none(), st.sampled_from(["a", "b", "c", "d", " a"])), min_size=1, max_size=20),
    mapping=st.dictionaries(st.sampled_from(["a", "b", "c", "x"]), st.one_of(st.none(), st.sampled_from(["A", "Z", ""])), max_size=4),
)
def test_every_row_gets_the_table_value_of_its_original_or_the_codes_value(values, mapping):
    column = pd.Series(values, dtype=object)
    code = wrap_with_value_map(UPPER_CODE, [{"from_value": k, "to_value": v} for k, v in mapping.items()])

    result = _run_in_process(code, column).tolist()

    for original, got in zip(values, result):
        if original is not None and original in mapping:
            assert pd.isna(got) if mapping[original] is None else got == mapping[original]
        elif original is None:
            assert got is None or pd.isna(got)
        else:
            assert got == original.upper()
