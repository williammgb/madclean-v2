"""Properties of the run records: the report's dictionary form, the token counters and the scores."""

import json
from dataclasses import asdict, fields, is_dataclass

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from fake_llm import FakeLLMClient
from madclean.evaluation import Evaluator
from madclean.evaluation.scoring import count_scores
from madclean.components.domain.report import (
    AgentTokenUsage,
    CleaningReport,
    ColumnReport,
    FDReport,
    SkippedColumnReport,
    TokenUsage,
    TraceEvent,
    TraceStep,
)
from madclean.config.loader import load_default_cleaning_config
from madclean.evaluation.scores import Scores
from madclean.llm.llm_registry import LLM_CLIENT_MAP, LLMSpec
from madclean.pipeline import Pipeline

AGENTS = ("recommender", "coding", "validation")
# A run field of the same name wins over an entry of that name, so the generator leaves them out.
RUN_FIELDS = ("token_usage", "cancelled", "runtime_seconds", "total_usage")

entry_keys = st.one_of(
    st.text(min_size=1, max_size=6),
    st.sampled_from(["café", "code → city", "住所", "a b"]),
).filter(lambda key: key not in RUN_FIELDS)

short_text = st.text(max_size=8)
generated_code = st.one_of(st.none(), st.just(""), st.text(max_size=20))
reasons = st.one_of(st.none(), st.text(min_size=1, max_size=20))
trace_steps = st.lists(st.builds(TraceStep, short_text, short_text, short_text, short_text), max_size=3)
counts = st.integers(min_value=0, max_value=2**40)
token_usage = st.builds(TokenUsage, counts, counts, counts)

column_reports = st.builds(
    ColumnReport,
    datatype=st.sampled_from(["DISCRETE_STRING", "INTEGER", "NATURAL_LANGUAGE_TEXT"]),
    already_clean=st.booleans(),
    cleaned=st.booleans(),
    attempts=st.integers(min_value=0, max_value=6),
    generated_code=generated_code,
    cleaning_validated=st.booleans(),
    trace_steps=trace_steps,
    reason=reasons,
)
fd_reports = st.builds(
    FDReport,
    target_columns=st.lists(short_text, max_size=3),
    cleaned=st.booleans(),
    attempts=st.integers(min_value=0, max_value=6),
    cleaning_validated=st.booleans(),
    generated_code=generated_code,
    trace_steps=trace_steps,
    reason=reasons,
)
skipped_reports = st.builds(
    SkippedColumnReport,
    datatype=st.sampled_from(["EMPTY", "UNKNOWN"]),
    already_clean=st.booleans(),
    cleaned=st.booleans(),
    attempts=st.integers(min_value=0, max_value=6),
    cleaning_validated=st.booleans(),
    reason=st.text(max_size=20),
)
reports = st.builds(
    CleaningReport,
    entries=st.dictionaries(entry_keys, st.one_of(column_reports, fd_reports, skipped_reports), max_size=4),
    token_usage=st.builds(AgentTokenUsage, token_usage, token_usage, token_usage),
    cancelled=st.booleans(),
    runtime_seconds=st.one_of(st.none(), st.floats(min_value=0, max_value=1e6, allow_nan=False)),
    total_usage=st.one_of(st.none(), token_usage),
)


@given(reports)
def test_the_report_dict_keeps_the_shape_the_thesis_wrote(report):
    data = report.to_dict()

    assert data == report.to_dict(), "two calls must give the same dictionary"
    assert json.loads(json.dumps(data)) == data, "the report must survive a round trip through JSON"

    expected_keys = list(report.entries) + ["token_usage"]
    if report.cancelled:
        expected_keys.append("cancelled")
    if report.runtime_seconds is not None:
        expected_keys.append("runtime_seconds")
    if report.total_usage is not None:
        expected_keys.append("total_usage")
    assert list(data) == expected_keys
    assert ("cancelled" in data) is report.cancelled
    assert ("runtime_seconds" in data) is (report.runtime_seconds is not None)
    assert ("total_usage" in data) is (report.total_usage is not None)
    assert list(data["token_usage"]) == list(AGENTS)

    for key, entry in report.entries.items():
        line = data[key]
        written = [f.name for f in fields(entry) if not (f.name == "reason" and entry.reason is None)]
        assert list(line) == written, "entry keys follow the field order"
        assert ("reason" in line) is (entry.reason is not None)
        if isinstance(entry, SkippedColumnReport):
            assert "generated_code" not in line and "trace_steps" not in line
        else:
            assert "generated_code" in line and "trace_steps" in line


@given(st.lists(st.tuples(st.sampled_from(AGENTS), counts, counts), max_size=20))
def test_token_counts_stay_the_sum_of_what_was_added(additions):
    usage = AgentTokenUsage()
    added = {agent: [0, 0] for agent in AGENTS}

    for agent, input_tokens, output_tokens in additions:
        getattr(usage, agent).add(input_tokens, output_tokens)
        added[agent][0] += input_tokens
        added[agent][1] += output_tokens
        counted = getattr(usage, agent)
        assert counted.total_tokens == counted.input_tokens + counted.output_tokens

    for agent, (input_tokens, output_tokens) in added.items():
        counted = getattr(usage, agent)
        assert (counted.input_tokens, counted.output_tokens) == (input_tokens, output_tokens)

    total = usage.total()
    assert total.input_tokens == sum(pair[0] for pair in added.values())
    assert total.output_tokens == sum(pair[1] for pair in added.values())
    assert total.total_tokens == total.input_tokens + total.output_tokens


SECTION_KEYS = {
    "detection_counts": [
        "true_positives", "false_positives", "false_negatives",
        "true_negatives", "total_errors", "total_changes",
    ],
    "detection_metrics": ["precision", "recall", "f1_score"],
    "correction_counts": [
        "correctly_repaired_cells", "incorrectly_repaired_cells", "repaired_clean_cells",
    ],
    "correction_metrics": ["precision", "recall", "f1_score"],
}


@st.composite
def mask_triples(draw):
    """Three masks of one shape: all false, all true, or drawn cell by cell."""
    rows = draw(st.integers(min_value=0, max_value=30))
    width = draw(st.integers(min_value=1, max_value=4))
    names = [f"c{index}" for index in range(width)]

    def one_mask():
        kind = draw(st.sampled_from(["none", "all", "mixed"]))
        if kind == "none":
            cells = [[False] * width for _ in range(rows)]
        elif kind == "all":
            cells = [[True] * width for _ in range(rows)]
        else:
            row = st.lists(st.booleans(), min_size=width, max_size=width)
            cells = draw(st.lists(row, min_size=rows, max_size=rows))
        if not rows:
            return pd.DataFrame([], columns=names, dtype=bool)
        return pd.DataFrame(cells, columns=names, dtype=bool)

    return one_mask(), one_mask(), one_mask()


@given(mask_triples())
def test_scores_keep_the_committed_key_structure_and_plain_number_types(masks):
    errors_mask, changes_mask, equal_mask = masks

    # The counting reads nothing but its three masks, so it needs no loaded dataset.
    scores = count_scores(errors_mask, changes_mask, equal_mask, errors_mask.size)
    data = asdict(scores)

    assert list(data) == list(SECTION_KEYS)
    json.dumps(data)
    for section, keys in SECTION_KEYS.items():
        assert list(data[section]) == keys
        wanted = int if section.endswith("counts") else float
        for key, value in data[section].items():
            assert type(value) is wanted, (section, key, type(value))
    for section in ("detection_metrics", "correction_metrics"):
        for key, value in data[section].items():
            assert 0.0 <= value <= 1.0, (section, key, value)


def test_a_run_hands_back_dataclasses_and_not_dicts(tmp_path):
    """The records the pipeline, the registry and the scorer produce are all dataclasses."""
    frame = pd.DataFrame({
        "letters": ["anna ", "Bram", "cees", "Dirk ", "eva", "Femke", "gijs", "Hanna"],
        "numbers": [10, 11, 12, 13, 14, 15, 16, 17],
        "blank": [None] * 8,
    })
    dirty_path = tmp_path / "small_dirty.csv"
    frame.to_csv(dirty_path, index=False)
    frame.to_csv(tmp_path / "small_gt.csv", index=False)

    fake = FakeLLMClient()
    events = []
    config = load_default_cleaning_config()
    config.verbose = False
    config.sampling_seed = 7
    config.enable_multi_col_cleaning = False
    llm = fake.llm_config()
    pipeline = Pipeline(
        llm_config=llm,
        agent_llm_configs={"recommender": llm, "coding": llm, "validation": llm},
        config=config,
        trace_callback=events.append,
        verbose=False,
    )
    cleaned, report = pipeline.run(file_path=str(dirty_path))

    assert is_dataclass(report) and isinstance(report, CleaningReport)
    assert isinstance(report.token_usage, AgentTokenUsage)
    assert isinstance(report.token_usage.recommender, TokenUsage)
    assert isinstance(report.total_usage, TokenUsage)
    assert report.entries, "the run must have reported on its columns"
    for name, entry in report.entries.items():
        assert is_dataclass(entry), (name, type(entry))
        for step in getattr(entry, "trace_steps", []):
            assert isinstance(step, TraceStep), (name, type(step))
    assert events and all(isinstance(event, TraceEvent) for event in events)
    assert all(isinstance(spec, LLMSpec) for spec in LLM_CLIENT_MAP.values())

    evaluator = Evaluator(dirty_path, tmp_path / "small_gt.csv", numeric_columns={"numbers"})
    evaluation = evaluator.evaluate(cleaned)
    assert isinstance(evaluation.overall, Scores)
    assert evaluation.per_column and all(
        isinstance(scores, Scores) for scores in evaluation.per_column.values()
    )
