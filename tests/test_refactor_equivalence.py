"""Records what a scripted run produces, so the slice 2 refactor can be proved to change nothing.

Each scenario drives the pipeline into a named set of branches with a scripted fake model, asserts
that it really reached them, and then compares the cleaned table, the run report, the trace events
per column, the prompt hashes per agent and the review payloads against a stored reference file.
A missing reference is written and the test fails with "reference written", so references are only
ever captured deliberately, on unchanged code.
"""

import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd
import pytest

from fake_llm import IDENTITY_CODE, FakeLLMClient, ServiceUnavailable, first_user_text
from madclean.components.coordinator.prompt_generation import PromptGeneration
from madclean.components.dataprofiler.semantic_mapping import SemanticTypeDetection
from madclean.components.multi_agent_cleaner.llm_coding import LLMCodingAgent
from madclean.config.loader import load_default_cleaning_config
from madclean.pipeline import Pipeline

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "refactor"
BEERS = Path(__file__).resolve().parents[1] / "data" / "benchmark_datasets" / "beers_dirty.csv"

FD_KEY = "codex → cityx"

# The sample sizes every reference was recorded with. The defaults have since moved to 500 dirty
# values; the scenarios keep these so they still compare against the same recorded runs.
RECORDED_SAMPLE_SIZES = {
    "NUMERIC": {"clean_sample_size": 50, "dirty_sample_size": 250},
    "DATETIME": {"clean_sample_size": 100, "dirty_sample_size": 250},
    "DIRTY_NUMERIC": {"random_sample_size": 50, "unique_sample_size": 250},
    "STRING": {"random_sample_size": 150, "unique_sample_size": 250},
    "NLT": {"short_sample_size": 100, "long_sample_size": 20},
}


# --------------------------------------------------------------------------------------
# Scripted answers
# --------------------------------------------------------------------------------------

CLEAN_RECOMMENDATION = json.dumps({
    "is_clean": True,
    "summary": "Column is already clean.",
    "error_types": [],
    "examples_clean": [],
    "examples_dirty": [],
    "cleaning_instructions": [],
})

NOT_JSON = "I am afraid I cannot answer that."

UPPER_CODE = (
    "import pandas as pd\n"
    "\n"
    "def clean_column(data):\n"
    "    if isinstance(data, pd.DataFrame):\n"
    "        return data.apply(lambda s: s.str.upper() if s.dtype == object else s)\n"
    "    return data.str.upper() if data.dtype == object else data\n"
)

BROKEN_CODE = "def clean_column(data):\n    raise ValueError('boom')\n"


def code_answer(code: str) -> str:
    return f"```python\n{code}```"


def validator_answer(needs_correction: bool, target: str | None = None, instructions: str = "") -> str:
    return json.dumps({
        "needs_correction": needs_correction,
        "feedback_target": target,
        "correction_instructions": instructions,
    })


class Script:
    """Answers per (agent, column) in call order; the last answer for a key repeats.

    Counting per column rather than per run keeps answers independent of how concurrent
    columns interleave, and a column's own calls are always sequential.
    """

    def __init__(self, answers: dict, columns, fd_key: str | None = None):
        self.answers = answers
        self.columns = tuple(columns)
        self.fd_key = fd_key
        self.fd_tokens = tuple(fd_key.split(" → ")) if fd_key else ()
        self.seen: Counter = Counter()

    def key_for(self, messages) -> str:
        text = first_user_text(messages)
        if self.fd_tokens and all(token in text for token in self.fd_tokens):
            return self.fd_key
        for column in self.columns:
            if column in text:
                return column
        return ""

    def __call__(self, kind, messages):
        key = (kind, self.key_for(messages))
        answers = self.answers.get(key)
        if not answers:
            return None
        index = min(self.seen[key], len(answers) - 1)
        self.seen[key] += 1
        return answers[index]


# --------------------------------------------------------------------------------------
# Running a scenario
# --------------------------------------------------------------------------------------


async def _execute_in_process(code_str, df, columns, timeout=30):
    """Stand-in for the subprocess runner: same input copy, same type check, same error shape."""
    input_df = df[[columns]].copy() if isinstance(columns, str) else df[list(columns)].copy()
    data = input_df[columns]
    expected_type = pd.Series if isinstance(columns, str) else pd.DataFrame
    expected_name = "Series" if isinstance(columns, str) else "DataFrame"
    namespace: dict = {}
    try:
        exec(code_str, namespace)
        clean_column = namespace.get("clean_column")
        if clean_column is None:
            raise NameError("LLM code did not define the 'clean_column' function.")
        result = clean_column(data)
        if not isinstance(result, expected_type):
            raise TypeError(f"clean_column must return a pandas {expected_name}, but returned {type(result).__name__}.")
        return result
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


@pytest.fixture(scope="module")
def shared_spacy_model():
    """One spaCy load for the whole module; the fast scenarios all reuse it."""
    return SemanticTypeDetection(verbose=False).nlp_model


@pytest.fixture
def fast_env(monkeypatch, shared_spacy_model):
    """Keeps the fast scenarios under a second each: no spaCy reload, no subprocess per code run."""
    monkeypatch.setattr(SemanticTypeDetection, "_load_spacy_model", lambda self: shared_spacy_model)
    monkeypatch.setattr(LLMCodingAgent, "_execute_code_async", staticmethod(_execute_in_process))


class Recorder:
    """Collects everything the pipeline hands to the GUI: trace events and review requests."""

    def __init__(self, user_validation=None, hitl=None):
        self.events: dict[str, list] = {}
        self.reviews: dict[str, list] = {}
        self._user_validation = user_validation
        self._hitl = hitl

    def trace(self, event):
        data = _plain(event)
        self.events.setdefault(str(data.get("column", "")), []).append(data)

    def user_validation(self, payload):
        self._store(f"user_validation::{payload.get('column', '')}", payload)
        return self._user_validation(payload)

    def hitl(self, payload):
        self._store(f"{payload.get('kind', '')}::{payload.get('column', '')}", payload)
        return self._hitl(payload)

    def _store(self, key, payload):
        # request_id holds time.time(), so it can never match a reference.
        self.reviews.setdefault(key, []).append(
            {name: _plain(value) for name, value in payload.items() if name != "request_id"}
        )


def _plain(value):
    """Records compare as plain JSON data, whether the pipeline yields dicts or dataclasses."""
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _report_dict(report) -> dict:
    data = report.to_dict() if hasattr(report, "to_dict") else report
    return {key: _plain(value) for key, value in data.items() if key != "runtime_seconds"}


def _table(cleaned: pd.DataFrame | None, digest: bool = False) -> dict:
    if cleaned is None:
        return {"csv": None, "dtypes": {}}
    # The references were recorded on Windows, where pandas ends lines with \r\n; naming the ending
    # makes Linux and macOS produce the same text.
    csv_text = cleaned.to_csv(index=False, lineterminator="\r\n")
    return {
        "csv": hashlib.sha256(csv_text.encode("utf-8")).hexdigest() if digest else csv_text,
        "dtypes": {str(name): str(dtype) for name, dtype in cleaned.dtypes.items()},
    }


def _prompt_hashes(fake: FakeLLMClient) -> dict:
    kinds = sorted({kind for kind, _ in fake.calls})
    return {
        kind: sorted(
            hashlib.sha256(json.dumps(messages, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            for messages in fake.calls_for(kind)
        )
        for kind in kinds
    }


def run_scenario(
    tmp_path: Path,
    frame: pd.DataFrame,
    *,
    script=None,
    coder_code: str = IDENTITY_CODE,
    user_validation=None,
    hitl=None,
    file_path: Path | None = None,
    **overrides,
) -> tuple[dict, dict]:
    """Runs one scripted pipeline run and returns (recording, report as a plain dict)."""
    if file_path is None:
        file_path = tmp_path / "scenario.csv"
        frame.to_csv(file_path, index=False)

    fake = FakeLLMClient(coder_code=coder_code, script=script)
    recorder = Recorder(user_validation, hitl)
    config = load_default_cleaning_config()
    config.verbose = False
    config.sampling_seed = 7
    config.sample_sizes = copy.deepcopy(RECORDED_SAMPLE_SIZES)
    for name, value in overrides.items():
        if not hasattr(config, name):
            raise AssertionError(f"unknown config field: {name}")
        setattr(config, name, value)

    llm = fake.llm_config()
    pipeline = Pipeline(
        llm_config=llm,
        agent_llm_configs={"recommender": llm, "coding": llm, "validation": llm},
        config=config,
        trace_callback=recorder.trace,
        user_validation_callback=recorder.user_validation if user_validation else None,
        hitl_callback=recorder.hitl if hitl else None,
        verbose=False,
    )
    cleaned, report = pipeline.run(file_path=str(file_path), save_cleaned=False)

    report_dict = _report_dict(report)
    recording = {
        "table": _table(cleaned, digest=file_path == BEERS),
        "report": report_dict,
        "traces": recorder.events,
        "prompts": _prompt_hashes(fake),
        "reviews": recorder.reviews,
    }
    return recording, report_dict


def compare_with_reference(name: str, recording: dict):
    path = GOLDEN_DIR / f"{name}.json"
    payload = json.loads(json.dumps(recording, ensure_ascii=False, default=str))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        pytest.fail(f"reference written: {path.name} — re-run to compare against it")
    expected = json.loads(path.read_text(encoding="utf-8"))
    for section in ("table", "report", "traces", "prompts", "reviews"):
        assert payload[section] == expected[section], f"{name}: {section} differs from the reference"


# --------------------------------------------------------------------------------------
# Test data
# --------------------------------------------------------------------------------------

CODES = ["A1", "A1", "A1", "B2", "B2", "B2", "C3", "C3", "C3", "D4",
         "D4", "D4", "E5", "E5", "E5", "F6", "F6", "F6", "G7", "G7"]
CITIES = ["Delft", "Delft", "Delft", "Gouda", "Gouda", "Gouda", "Breda", "Breda", "Breda", "Emmen",
          "Emmen", "Emmen", "Almere", "Almere", "Lelystad", "Venlo", "Venlo", None, "Zwolle", "Zwolle"]
NAMES = ["anna ", "Bram", "cees", "Dirk ", "eva", "Femke", "gijs", "Hanna", "ivo", "Joost",
         "Kim ", "lars", "Mila", "noor", "Olaf", "pim", "Quinn", "roos", "Sam", "tess"]


def dependency_columns() -> dict:
    """A dependency codex → cityx with one conflicting value and one missing value."""
    return {"codex": list(CODES), "cityx": list(CITIES)}


def string_column(prefix: str) -> list[str]:
    """Single-token values, so each column's name appears in its own prompts and nowhere else."""
    return [f"{prefix}_{name}" for name in NAMES]


# --------------------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------------------


def test_defaults(tmp_path, fast_env):
    """String validated, numeric validation skipped, already clean, EMPTY, skipped, one FD task."""
    frame = pd.DataFrame({
        "colstr": string_column("colstr"),
        "colnum": [10 + i for i in range(20)],
        "colclean": string_column("colclean"),
        "colskip": string_column("colskip"),
        "colempty": [None] * 20,
        **dependency_columns(),
    })
    script = Script(
        {("recommender", "colclean"): [CLEAN_RECOMMENDATION]},
        columns=("colstr", "colnum", "colclean", "colskip", "colempty", "codex", "cityx"),
        fd_key=FD_KEY,
    )

    recording, report = run_scenario(tmp_path, frame, script=script, skip_columns=["colskip"])

    assert report["colstr"] == {
        "datatype": "DISCRETE_STRING", "already_clean": False, "cleaned": True, "attempts": 1,
        "generated_code": IDENTITY_CODE.strip(), "cleaning_validated": True,
        "trace_steps": report["colstr"]["trace_steps"],
    }
    assert [step["id"] for step in report["colstr"]["trace_steps"]] == ["recommender_1", "coder_1", "validator_1"]
    assert report["colnum"]["cleaning_validated"] is False and report["colnum"]["cleaned"] is True
    assert report["colclean"]["already_clean"] is True and report["colclean"]["cleaned"] is False
    assert report["colskip"] == {
        "datatype": "DISCRETE_STRING", "already_clean": True, "cleaned": False, "attempts": 0,
        "generated_code": "", "cleaning_validated": False,
        "trace_steps": [{"id": "finished", "title": "Finished", "status": "completed",
                         "output": "User has determined this column is already clean."}],
    }
    assert report["colempty"] == {
        "datatype": "EMPTY", "already_clean": False, "cleaned": False, "attempts": 0,
        "cleaning_validated": False, "reason": "Empty column or unknown data type",
    }
    assert report[FD_KEY]["cleaned"] is True and report[FD_KEY]["cleaning_validated"] is False
    assert report[FD_KEY]["target_columns"] == ["codex", "cityx"]

    compare_with_reference("defaults", recording)


def test_feedback_loops(tmp_path, fast_env):
    """Validator feedback to coder then recommender, attempts running out, bad JSON, broken code, FD feedback."""
    frame = pd.DataFrame({
        "colcoder": string_column("colcoder"),
        "colreject": string_column("colreject"),
        "colbadjson": string_column("colbadjson"),
        "colbreak": string_column("colbreak"),
        **dependency_columns(),
    })
    script = Script(
        {
            ("validator", "colcoder"): [
                validator_answer(True, "CODER", "Keep the original spacing."),
                validator_answer(True, "RECOMMENDER", "Describe the format more precisely."),
                validator_answer(False),
            ],
            ("validator", "colreject"): [validator_answer(True, "RECOMMENDER", "Still wrong.")],
            # Three invalid answers exhaust the parse retries, so the loop itself asks again.
            ("recommender", "colbadjson"): [NOT_JSON, NOT_JSON, NOT_JSON, None],
            ("coder", "colbreak"): [code_answer(BROKEN_CODE), code_answer(IDENTITY_CODE)],
            ("validator", FD_KEY): [validator_answer(True, "CODER", "Impute the missing city."), validator_answer(False)],
        },
        columns=("colcoder", "colreject", "colbadjson", "colbreak", "codex", "cityx"),
        fd_key=FD_KEY,
    )

    recording, report = run_scenario(
        tmp_path, frame, script=script,
        max_cleaning_attempts=3, max_multi_col_attempts=2, max_coding_attempts=1,
        enable_validation_multi=True,
    )

    assert report["colcoder"]["attempts"] == 3 and report["colcoder"]["cleaned"] is True
    assert [step["status"] for step in report["colcoder"]["trace_steps"]].count("needs_correction") == 2
    assert report["colreject"]["cleaned"] is False and report["colreject"]["attempts"] == 3
    assert report["colbadjson"]["attempts"] == 2 and report["colbadjson"]["cleaned"] is True
    assert report["colbadjson"]["trace_steps"][0]["status"] == "invalid_response"
    assert report["colbreak"]["attempts"] == 2 and report["colbreak"]["cleaned"] is True
    assert report["colbreak"]["trace_steps"][1]["status"] == "failed"
    assert report[FD_KEY]["cleaned"] is True and report[FD_KEY]["cleaning_validated"] is True
    assert report[FD_KEY]["attempts"] == 2

    compare_with_reference("feedback_loops", recording)


def test_validator_leave_uncleaned(tmp_path, fast_env):
    """A validator that never answers leaves both the column and the FD task uncleaned."""
    frame = pd.DataFrame({"colstr": string_column("colstr"), **dependency_columns()})
    script = Script(
        {("validator", "colstr"): [NOT_JSON], ("validator", FD_KEY): [NOT_JSON]},
        columns=("colstr", "codex", "cityx"),
        fd_key=FD_KEY,
    )

    recording, report = run_scenario(
        tmp_path, frame, script=script, coder_code=UPPER_CODE,
        max_cleaning_attempts=2, max_multi_col_attempts=2, enable_validation_multi=True,
        validator_failure_strategy="leave_uncleaned",
    )

    assert report["colstr"]["cleaned"] is False
    assert report["colstr"]["reason"] == "Validator failed and strategy is leave_uncleaned"
    assert report[FD_KEY]["reason"] == "Validator failed and strategy is leave_uncleaned"
    # The column was left uncleaned, so the uppercase code did not reach the table.
    assert recording["table"]["csv"].splitlines()[1].startswith("colstr_anna")

    compare_with_reference("validator_leave_uncleaned", recording)


def test_validator_ask_user(tmp_path, fast_env):
    """A failing validator asks the user, who approves the cleaning."""
    frame = pd.DataFrame({"colstr": string_column("colstr")})
    script = Script({("validator", "colstr"): [NOT_JSON]}, columns=("colstr",))

    def approve(payload):
        return {"needs_correction": False}

    recording, report = run_scenario(
        tmp_path, frame, script=script, coder_code=UPPER_CODE, user_validation=approve,
        max_cleaning_attempts=2, validator_failure_strategy="ask_user", enable_multi_col_cleaning=False,
    )

    assert report["colstr"]["cleaned"] is True and report["colstr"]["cleaning_validated"] is True
    asked = recording["reviews"]["user_validation::colstr"]
    assert len(asked) == 1 and asked[0]["default_feedback"] == "Validator failed to validate. Please provide final feedback."
    assert asked[0]["modified_count"] == 20
    statuses = [event["status"] for event in recording["traces"]["colstr"]]
    assert "needs_user_validation" in statuses

    compare_with_reference("validator_ask_user", recording)


def test_user_validation(tmp_path, fast_env):
    """The user replaces the validator: feedback to coder, then recommender, then approval."""
    frame = pd.DataFrame({
        "colcoder": string_column("colcoder"),
        "colfail": string_column("colfail"),
        **dependency_columns(),
    })
    decisions = {
        "colcoder": [
            {"needs_correction": True, "feedback_target": "CODER", "correction_instructions": "Keep the spacing."},
            {"needs_correction": True, "feedback_target": "RECOMMENDER", "correction_instructions": "Be precise."},
            {"needs_correction": False},
        ],
        "colfail": [{"needs_correction": True, "feedback_target": "RECOMMENDER", "correction_instructions": "No."}],
        FD_KEY: [{"needs_correction": False}],
    }
    seen: Counter = Counter()

    def decide(payload):
        column = payload["column"]
        # The two dependency columns are deliberately absent, so this raises for them. The
        # pipeline swallows a failing callback and treats it as "needs correction", which is
        # the branch the reference pins for codex and cityx.
        answers = decisions[column]
        index = min(seen[column], len(answers) - 1)
        seen[column] += 1
        return answers[index]

    recording, report = run_scenario(
        tmp_path, frame, coder_code=UPPER_CODE, user_validation=decide,
        max_cleaning_attempts=3, max_multi_col_attempts=2,
        enable_user_validation=True, enable_validation_multi=True,
    )

    assert report["colcoder"]["attempts"] == 3 and report["colcoder"]["cleaning_validated"] is True
    assert report["colfail"]["cleaned"] is False and report["colfail"]["attempts"] == 3
    assert report[FD_KEY]["cleaned"] is True and report[FD_KEY]["cleaning_validated"] is True
    assert len(recording["reviews"]["user_validation::colcoder"]) == 3
    assert recording["reviews"]["user_validation::colcoder"][0]["sample_rows"], "the review sample must not be empty"

    compare_with_reference("user_validation", recording)


def test_human_review(tmp_path, fast_env):
    """Human in the loop: already-clean review, code review, and the validation-review decisions."""
    frame = pd.DataFrame({
        "colclean": string_column("colclean"),
        "colgood": string_column("colgood"),
        "colbad": string_column("colbad"),
        "coldisagree": string_column("coldisagree"),
        "colrejected": string_column("colrejected"),
    })
    script = Script(
        {
            ("recommender", "colclean"): [CLEAN_RECOMMENDATION],
            ("validator", "colrejected"): [validator_answer(True, "RECOMMENDER", "Trailing spaces remain.")],
        },
        columns=("colclean", "colgood", "colbad", "coldisagree", "colrejected"),
    )
    review_seen: Counter = Counter()

    def review(payload):
        kind, column = payload["kind"], payload["column"]
        index = review_seen[(kind, column)]
        review_seen[(kind, column)] += 1
        if kind == "already_clean_review":
            if index == 0:
                return {"decision": "reject", "rejection_reason": "Some values still have trailing spaces."}
            return {"decision": "confirm"}
        if kind == "code_review":
            if column == "colgood":
                return {"code": UPPER_CODE}
            if column == "colbad":
                return {"code": BROKEN_CODE}
            return {"code": payload["code"]}
        if column == "coldisagree":
            if index == 0:
                return {"decision": "validator_ok_disagree", "feedback_target": "RECOMMENDER",
                        "correction_instructions": "The casing is still wrong."}
            return {"decision": "validator_ok_agree"}
        if column == "colrejected":
            return {"decision": "feedback_reject_cleaning_valid"}
        return {"decision": "validator_ok_agree"}

    recording, report = run_scenario(
        tmp_path, frame, script=script, hitl=review,
        max_cleaning_attempts=3, enable_multi_col_cleaning=False, human_in_the_loop=True,
    )

    assert report["colclean"]["already_clean"] is True and report["colclean"]["attempts"] == 2
    assert report["colclean"]["cleaning_validated"] is True
    assert [step["status"] for step in report["colclean"]["trace_steps"]] == [
        "already_clean", "rejected", "already_clean", "approved",
    ]
    # The accepted edit ran and reached the table; the broken edit was dropped for the coder's own code.
    assert recording["table"]["csv"].splitlines()[1].split(",")[1] == "COLGOOD_ANNA "
    assert recording["table"]["csv"].splitlines()[1].split(",")[2] == "colbad_anna "
    assert any(event["title"] == "User code review" for event in recording["traces"]["colbad"])
    assert report["coldisagree"]["attempts"] == 2 and report["coldisagree"]["cleaned"] is True
    assert report["colrejected"]["cleaned"] is True and report["colrejected"]["cleaning_validated"] is True

    compare_with_reference("human_review", recording)


def test_no_validation_api_errors(tmp_path, fast_env, monkeypatch):
    """Validation off, a provider that is unavailable for three agents, and a task that raises."""
    frame = pd.DataFrame({
        "colplain": string_column("colplain"),
        "colrec503": string_column("colrec503"),
        "colcode503": string_column("colcode503"),
        "colboom": string_column("colboom"),
        **dependency_columns(),
    })
    script = Script(
        {
            ("recommender", "colrec503"): [ServiceUnavailable("provider unavailable")],
            ("coder", "colcode503"): [ServiceUnavailable("provider unavailable")],
            ("fd_recommender", FD_KEY): [ServiceUnavailable("provider unavailable")],
        },
        columns=("colplain", "colrec503", "colcode503", "colboom", "codex", "cityx"),
        fd_key=FD_KEY,
    )
    original_prompt = PromptGeneration.create_prompt_recommender

    def raise_for_colboom(self, col, profile, **kwargs):
        if col == "colboom":
            raise RuntimeError("prompt creation broke for colboom")
        return original_prompt(self, col, profile, **kwargs)

    monkeypatch.setattr(PromptGeneration, "create_prompt_recommender", raise_for_colboom)

    recording, report = run_scenario(
        tmp_path, frame, script=script, coder_code=UPPER_CODE,
        enable_validation=False, max_cleaning_attempts=2, max_multi_col_attempts=2, max_coding_attempts=1,
    )

    assert report["colplain"]["cleaned"] is True and report["colplain"]["cleaning_validated"] is False
    assert report["colrec503"]["cleaned"] is False
    assert [step["status"] for step in report["colrec503"]["trace_steps"]] == ["api_unavailable", "api_unavailable"]
    assert report["colcode503"]["cleaned"] is False
    assert "api_unavailable" in [step["status"] for step in report["colcode503"]["trace_steps"]]
    assert report["colboom"]["cleaned"] is False
    assert report["colboom"]["reason"] == "RuntimeError: prompt creation broke for colboom"
    assert report[FD_KEY]["cleaned"] is False
    assert recording["traces"][FD_KEY][-1]["status"] == "failed"

    compare_with_reference("no_validation_api_errors", recording)


@pytest.mark.slow
def test_beers(tmp_path):
    """The full beers run, with real subprocess execution and a real spaCy load."""
    recording, report = run_scenario(tmp_path, frame=None, file_path=BEERS)

    assert report["total_usage"]["total_tokens"] > 0
    assert report["brewery_id"]["cleaned"] is True
    assert len(recording["traces"]) >= len(pd.read_csv(BEERS).columns)

    compare_with_reference("beers", recording)
