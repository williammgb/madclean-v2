"""The notebook a run hands you has to run, and has to give back the same table.

The test does what a person would: it takes a real fake-model run, exports the notebook, then
executes its code cells in order in a fresh namespace — no Jupyter, no model — and compares the
CSV the notebook wrote against the table the run produced.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from fake_llm import FakeLLMClient
from madclean.config.loader import load_default_cleaning_config
from madclean.notebook import build_notebook, merged_cleaning_code, notebook_json
from madclean.pipeline import Pipeline

# A cleaner that actually changes something, so the notebook has work to reproduce.
STRIP_CODE = """
import pandas as pd

def clean_column(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.title()
"""


def _run(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "letters": ["anna ", "BRAM", " cees", "dirk ", "eva", "femke", "gijs", "hanna"],
            "numbers": [10, 11, 12, 13, 14, 15, 16, 17],
        }
    )
    dirty = tmp_path / "small_dirty.csv"
    frame.to_csv(dirty, index=False)

    fake = FakeLLMClient(coder_code=STRIP_CODE.strip())
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
    cleaned, report = pipeline.run(file_path=str(dirty))
    return dirty, cleaned, report


def _execute(notebook: dict, workdir: Path) -> dict:
    """Runs every code cell in order, as opening the notebook and pressing run-all would."""
    namespace: dict = {"__name__": "__main__"}
    cwd = Path.cwd()
    try:
        import os

        os.chdir(workdir)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            exec("".join(cell["source"]), namespace)  # noqa: S102 - running the export is the test
    finally:
        import os

        os.chdir(cwd)
    return namespace


def test_the_notebook_runs_top_to_bottom_and_reproduces_the_cleaned_table(tmp_path):
    dirty, cleaned, report = _run(tmp_path)

    notebook = build_notebook(report, dataset_path=str(dirty), output_path="from_notebook.csv")
    namespace = _execute(notebook, tmp_path)

    written = tmp_path / "from_notebook.csv"
    assert written.exists(), "the notebook never wrote its cleaned file"

    from_notebook = pd.read_csv(written)
    from_run = cleaned.reset_index(drop=True)

    assert list(from_notebook.columns) == list(from_run.columns)
    pd.testing.assert_frame_equal(from_notebook.astype(str), from_run.astype(str))
    assert "cleaned" in namespace


def test_the_notebook_calls_no_model(tmp_path):
    """Whoever opens it can run it for nothing: nothing in it reaches for a model or a key."""
    dirty, _cleaned, report = _run(tmp_path)

    source = notebook_json(report, dataset_path=str(dirty))

    for forbidden in ("madclean", "openai", "google", "API_KEY", "llm", "Pipeline"):
        assert forbidden not in source, f"the notebook mentions {forbidden}"


def test_it_is_a_valid_notebook_file(tmp_path):
    dirty, _cleaned, report = _run(tmp_path)

    notebook = json.loads(notebook_json(report, dataset_path=str(dirty)))

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    assert notebook["cells"], "a notebook with no cells is not a notebook"
    assert all(cell["cell_type"] in ("code", "markdown") for cell in notebook["cells"])
    assert all(isinstance(cell["source"], list) for cell in notebook["cells"])
    # It explains itself, rather than being a wall of code.
    assert any(cell["cell_type"] == "markdown" for cell in notebook["cells"])


def test_a_run_that_changed_nothing_still_gives_a_runnable_notebook(tmp_path):
    """Every column already clean means no cleaning code — the notebook must still run."""
    notebook = build_notebook({}, dataset_path="none.csv", output_path="out.csv")
    frame = pd.DataFrame({"a": [1, 2, 3]})
    frame.to_csv(tmp_path / "none.csv", index=False)

    namespace = _execute(notebook, tmp_path)

    assert (tmp_path / "out.csv").exists()
    pd.testing.assert_frame_equal(namespace["cleaned"].astype(str), frame.astype(str))


def test_the_gui_and_the_notebook_share_one_code_builder(tmp_path):
    """The code the Report view shows and the code in the notebook are built by the same function."""
    _dirty, _cleaned, report = _run(tmp_path)

    code = merged_cleaning_code(report)

    assert "def clean_dataset(" in code
    assert "import pandas as pd" in code
    # Every task in the code is also a cell in the notebook.
    notebook = build_notebook(report, dataset_path="x.csv")
    cell_text = "".join("".join(cell["source"]) for cell in notebook["cells"])
    for line in code.splitlines():
        if line.startswith("def clean_"):
            assert line in cell_text


VALUE_TABLE_ANSWER = json.dumps({
    "analysis": "Two spellings of one town.",
    "is_clean": False,
    "summary": "Town names.",
    "error_types": ["variant spelling"],
    "examples_clean": ["Delft"],
    "examples_dirty": ["Delft ZH → Delft"],
    "cleaning_instructions": None,
    "value_mapping": [{"from_value": "Delft ZH", "to_value": "Delft"}, {"from_value": "??", "to_value": None}],
})

DEPENDENCY_TABLE_ANSWER = json.dumps({
    "analysis": "E5 is Almere twice and Lelystad once.",
    "summary": "Each code belongs to one city.",
    "corrections": [{"lhs_value": "E5", "correct_rhs": "Almere"}],
    "skipped_lhs_values": None,
    "impute_missing": True,
})


def test_a_run_with_a_value_table_and_a_dependency_table_exports_a_notebook_that_reproduces_it(tmp_path):
    frame = pd.DataFrame({
        "town": ["Delft", "Delft ZH", "Gouda", "Delft", "??", "Breda", "Delft ZH", "Gouda"] * 2 + ["Gouda"] * 4,
        "codex": ["A1", "A1", "A1", "B2", "B2", "B2", "C3", "C3", "C3", "D4",
                  "D4", "D4", "E5", "E5", "E5", "F6", "F6", "F6", "G7", "G7"],
        "cityx": ["Delft", "Delft", "Delft", "Gouda", "Gouda", "Gouda", "Breda", "Breda", "Breda", "Emmen",
                  "Emmen", "Emmen", "Almere", "Almere", "Lelystad", "Venlo", "Venlo", None, "Zwolle", "Zwolle"],
    })
    dirty = tmp_path / "tables_dirty.csv"
    frame.to_csv(dirty, index=False)

    def script(kind, messages):
        text = next(m["content"] for m in messages if m["role"] == "user")
        if kind == "recommender" and "'town'" in text:
            return VALUE_TABLE_ANSWER
        if kind == "fd_recommender":
            return DEPENDENCY_TABLE_ANSWER
        return None

    fake = FakeLLMClient(script=script)
    config = load_default_cleaning_config()
    config.verbose = False
    config.sampling_seed = 7
    llm = fake.llm_config()
    pipeline = Pipeline(
        llm_config=llm,
        agent_llm_configs={"recommender": llm, "coding": llm, "validation": llm},
        config=config,
        verbose=False,
    )
    cleaned, report = pipeline.run(file_path=str(dirty), save_cleaned=False)

    # Both tables really did their work, so the notebook has something to reproduce.
    assert cleaned["town"].tolist().count("Delft ZH") == 0 and cleaned["town"].isna().sum() == 2
    assert cleaned["cityx"].tolist()[14:18] == ["Almere", "Venlo", "Venlo", "Venlo"]
    assert "_VALUE_MAP" in report.to_dict()["town"]["generated_code"]

    notebook = build_notebook(report, dataset_path=str(dirty), output_path="from_notebook.csv")
    _execute(notebook, tmp_path)

    from_notebook = pd.read_csv(tmp_path / "from_notebook.csv")
    pd.testing.assert_frame_equal(
        from_notebook.astype(str).reset_index(drop=True),
        cleaned.astype(str).reset_index(drop=True),
    )


@pytest.mark.slow
def test_the_beers_run_exports_a_notebook_that_reproduces_it(tmp_path):
    """The same check at real size, on the benchmark the thesis reports."""
    beers = Path(__file__).resolve().parents[1] / "data" / "benchmark_datasets" / "beers_dirty.csv"
    fake = FakeLLMClient(coder_code=STRIP_CODE.strip())
    config = load_default_cleaning_config()
    config.verbose = False
    config.sampling_seed = 7
    llm = fake.llm_config()
    pipeline = Pipeline(
        llm_config=llm,
        agent_llm_configs={"recommender": llm, "coding": llm, "validation": llm},
        config=config,
        verbose=False,
    )
    cleaned, report = pipeline.run(file_path=str(beers))

    notebook = build_notebook(report, dataset_path=str(beers), output_path="beers_from_notebook.csv")
    _execute(notebook, tmp_path)

    from_notebook = pd.read_csv(tmp_path / "beers_from_notebook.csv")
    pd.testing.assert_frame_equal(
        from_notebook.astype(str).reset_index(drop=True),
        cleaned.astype(str).reset_index(drop=True),
    )
