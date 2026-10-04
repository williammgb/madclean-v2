import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from fake_llm import FakeLLMClient
from madclean.components.coordinator.cleaning_coordinator import CleaningCoordinator
from madclean.components.coordinator.prompt_generation import PromptGeneration
from madclean.components.domain.report import AgentTokenUsage, CleaningReport
from madclean.components.domain.schema import ColumnProfile, FDResult, MultiColumnTask
from madclean.components.multi_agent_cleaner.llm_coding import LLMCodingAgent
from madclean.config.loader import load_default_cleaning_config
from madclean.pipeline import Pipeline
from madclean.utils.helpers import load_dataset

BEERS = Path(__file__).resolve().parents[1] / "data" / "benchmark_datasets" / "beers_dirty.csv"


@dataclass
class OfflineRun:
    fake: FakeLLMClient
    cleaned: pd.DataFrame | None
    report: CleaningReport | None
    saved: list[Path]
    seconds: float


def make_pipeline(fake: FakeLLMClient, **config_overrides) -> Pipeline:
    config = load_default_cleaning_config()
    config.verbose = False
    for name, value in config_overrides.items():
        setattr(config, name, value)
    llm = fake.llm_config()
    return Pipeline(
        llm_config=llm,
        agent_llm_configs={"recommender": llm, "coding": llm, "validation": llm},
        config=config,
        verbose=False,
    )


def run_offline(out_dir: Path, file_path: Path, fake: FakeLLMClient | None = None, **config_overrides) -> OfflineRun:
    fake = fake or FakeLLMClient()
    pipeline = make_pipeline(fake, **config_overrides)
    saved: list[Path] = []

    def save_into_out_dir(cleaned_df, input_path, base_dir):
        target = out_dir / f"{Path(input_path).stem}_cleaned.csv"
        cleaned_df.to_csv(target, index=False)
        saved.append(target)
        return target

    started = time.perf_counter()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("madclean.pipeline.save_dataset", save_into_out_dir)
        cleaned, report = pipeline.run(file_path=str(file_path), save_cleaned=True)
    return OfflineRun(fake, cleaned, report, saved, time.perf_counter() - started)


@pytest.fixture(scope="module")
def beers_run(tmp_path_factory):
    return run_offline(tmp_path_factory.mktemp("beers_a"), BEERS, sampling_seed=7)


def _prompts_by_agent(fake: FakeLLMClient) -> dict[str, list[str]]:
    kinds = sorted({kind for kind, _ in fake.calls})
    return {kind: sorted(repr(messages) for messages in fake.calls_for(kind)) for kind in kinds}


def _assert_same_run(first: OfflineRun, second: OfflineRun):
    pd.testing.assert_frame_equal(first.cleaned, second.cleaned)
    without_runtime = [
        {k: v for k, v in run.report.to_dict().items() if k != "runtime_seconds"} for run in (first, second)
    ]
    assert without_runtime[0] == without_runtime[1]
    assert first.fake.calls_for("validator")
    assert _prompts_by_agent(first.fake) == _prompts_by_agent(second.fake)


def test_two_runs_with_the_same_seed_on_a_beers_slice_send_the_same_prompts(tmp_path):
    # 300 rows of three redundant string/id columns: enough for sampling to choose and for dependency tasks.
    slice_csv = tmp_path / "beers_slice.csv"
    load_dataset(BEERS)[["brewery_id", "brewery_name", "city"]].head(300).to_csv(slice_csv, index=False)

    first = run_offline(tmp_path, slice_csv, sampling_seed=7)
    second = run_offline(tmp_path, slice_csv, sampling_seed=7)

    _assert_same_run(first, second)


@pytest.mark.slow
def test_two_full_beers_runs_with_the_same_seed_send_the_same_prompts(beers_run, tmp_path):
    second = run_offline(tmp_path, BEERS, sampling_seed=7)

    _assert_same_run(beers_run, second)


def test_fake_beers_run_cleans_every_column_and_keeps_the_data(beers_run):
    dirty = load_dataset(BEERS)
    cleaned, report = beers_run.cleaned, beers_run.report

    # ibu holds whole numbers and blanks, so it loads as decimals (45.0); the run writes it back whole.
    expected = dirty.assign(ibu=dirty["ibu"].astype("Int64"))
    pd.testing.assert_frame_equal(cleaned, expected)
    for column in dirty.columns:
        entry = report.entries[column]
        if entry.datatype in ("EMPTY", "UNKNOWN"):
            continue
        assert entry.cleaned is True, (column, entry)
    assert report.total_usage.total_tokens > 0
    assert beers_run.fake.calls_for("recommender")
    assert beers_run.fake.calls_for("coder")
    assert beers_run.fake.calls_for("validator")
    assert len(beers_run.saved) == 1
    print(f"fake beers run took {beers_run.seconds:.1f}s")


def _write_names_csv(path: Path) -> Path:
    names = ["Anna", "Bram", "Cees", "Dirk", "Eva", "Femke", "Gijs", "Hanna", "Ivo", "Joost"]
    pd.DataFrame({
        "a": names,
        "b": list(reversed(names)),
        "c": [f"{name} Jansen" for name in names],
    }).to_csv(path, index=False)
    return path


def test_one_column_that_raises_is_marked_failed_and_the_others_are_cleaned(tmp_path, monkeypatch):
    original = PromptGeneration.create_prompt_recommender

    def raise_for_column_b(self, col, profile, **kwargs):
        if col == "b":
            raise RuntimeError("prompt creation broke for b")
        return original(self, col, profile, **kwargs)

    monkeypatch.setattr(PromptGeneration, "create_prompt_recommender", raise_for_column_b)
    csv = _write_names_csv(tmp_path / "names.csv")
    fake = FakeLLMClient(coder_code="def clean_column(data):\n    return data.astype(str) + '_x'\n")

    run = run_offline(tmp_path, csv, fake, enable_multi_col_cleaning=False)

    dirty = pd.read_csv(csv)
    assert run.cleaned["a"].tolist() == [f"{v}_x" for v in dirty["a"]]
    assert run.cleaned["c"].tolist() == [f"{v}_x" for v in dirty["c"]]
    assert run.cleaned["b"].tolist() == dirty["b"].tolist()
    assert run.report.entries["a"].cleaned is True
    assert run.report.entries["c"].cleaned is True
    assert run.report.entries["b"].cleaned is False
    assert "RuntimeError: prompt creation broke for b" in run.report.entries["b"].reason


def test_a_stopped_run_returns_no_table_and_saves_nothing(tmp_path, monkeypatch):
    csv = _write_names_csv(tmp_path / "names.csv")

    def stop(self, df, column_profiles, multi_col_tasks):
        raise asyncio.CancelledError()

    monkeypatch.setattr(CleaningCoordinator, "clean_dataset", stop)
    run = run_offline(tmp_path, csv, enable_multi_col_cleaning=False)

    assert run.cleaned is None
    assert run.report.cancelled is True
    assert run.saved == []


@pytest.mark.slow
def test_ten_columns_run_at_least_twice_as_fast_as_with_the_thesis_blocking_execution(tmp_path, monkeypatch):
    rows = range(30)
    pd.DataFrame({f"n{i}": [100 + (r * (i + 3)) % 900 for r in rows] for i in range(10)}).to_csv(
        tmp_path / "numbers.csv", index=False
    )
    sleeping_code = "import time\n\ndef clean_column(data):\n    time.sleep(1)\n    return data\n"

    new = run_offline(tmp_path, tmp_path / "numbers.csv", FakeLLMClient(sleeping_code), enable_multi_col_cleaning=False)

    async def thesis_blocking_execution(code_str, df, columns, timeout=30):
        return LLMCodingAgent._execute_code(code_str, df, columns, timeout)

    monkeypatch.setattr(LLMCodingAgent, "_execute_code_async", staticmethod(thesis_blocking_execution))
    thesis = run_offline(tmp_path, tmp_path / "numbers.csv", FakeLLMClient(sleeping_code), enable_multi_col_cleaning=False)

    print(f"10 columns: {new.seconds:.1f}s with code off the event loop, {thesis.seconds:.1f}s with thesis blocking execution")
    assert len(new.fake.calls_for("coder")) >= 10
    assert len(thesis.fake.calls_for("coder")) >= 10
    assert all(new.report.entries[f"n{i}"].cleaned for i in range(10))
    assert new.seconds * 2 < thesis.seconds


class _StubLoop:
    """The parts of MultiAgentCleaning the coordinator touches; column cleaning returns values unchanged."""

    def __init__(self):
        self.cleaning_report = {}
        self.token_usage = AgentTokenUsage()
        self.per_task_usage = {}
        self.traces = []

    def reset_token_usage(self):
        self.token_usage = AgentTokenUsage()
        self.per_task_usage = {}

    def _emit_trace(self, event):
        self.traces.append(event)

    async def _run_column_cleaning_async(self, df, col, profile):
        self.cleaning_report[col] = {"cleaned": True}
        return col, df[col], f"[{col}] ok"


class _RaisingFDCleaner:
    task_type = "FD"

    def get_data(self, df, task_info):
        raise RuntimeError("dependency data broke")


def test_a_dependency_task_that_raises_is_marked_failed_and_the_run_returns():
    config = load_default_cleaning_config()
    config.verbose = False
    loop = _StubLoop()
    coordinator = CleaningCoordinator(loop, multi_column_cleaners=[_RaisingFDCleaner()], config=config)
    df = pd.DataFrame({"a": ["x", "x", "y"], "b": ["1", "1", "2"]})
    profiles = {c: ColumnProfile(name=c, semantic_type="DISCRETE_STRING", sample="") for c in df.columns}
    task = MultiColumnTask(task_type="FD", target_columns=["a", "b"], verbose_key="a → b", data=FDResult("a", "b", 1.0))

    cleaned, report = coordinator.clean_dataset(df, profiles, [task])

    pd.testing.assert_frame_equal(cleaned, df)
    assert report.entries["a → b"].cleaned is False
    assert report.entries["a → b"].target_columns == ["a", "b"]
    assert "RuntimeError: dependency data broke" in report.entries["a → b"].reason
    assert loop.traces[-1].status == "failed"
