"""Turning one cleaning run into a notebook that reproduces it without a model.

The agents' work ends as code: one function per column or dependency task. That code is the whole
result, so a run can be handed over as a notebook that loads the dirty file, runs those functions
in the order the run ran them, and saves the cleaned file. Nothing in the notebook calls a model,
so it runs anywhere, costs nothing and gives the same answer every time.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any

# Cells are plain Jupyter v4; no notebook library is needed to write one.
NOTEBOOK_FORMAT = {"nbformat": 4, "nbformat_minor": 5}
KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}


def _safe_name(key: str) -> str:
    """A function name from a task key: "codex → cityx" becomes "codex_cityx"."""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", key).strip("_").lower() or "task"


def _as_report_dict(report: Any) -> dict:
    """Accepts the run's report record or the dictionary form the GUI keeps."""
    if is_dataclass(report) and not isinstance(report, type):
        to_dict = getattr(report, "to_dict", None)
        return to_dict() if callable(to_dict) else asdict(report)
    return dict(report or {})


def cleaning_functions(report: Any) -> tuple[list[str], list[str], list[str]]:
    """The imports, the functions and the calls that make up one run's cleaning code.

    Tasks that were already clean, that failed, or that produced no code are left out: there is
    nothing to reproduce for them.
    """
    data = _as_report_dict(report)
    imports: list[str] = []
    functions: list[str] = []
    calls: list[str] = []

    entries = [(key, value) for key, value in data.items() if isinstance(value, dict)]
    columns = [(k, v) for k, v in entries if not v.get("target_columns")]
    dependencies = [(k, v) for k, v in entries if v.get("target_columns")]

    for key, entry in columns + dependencies:
        if key == "ALL_MERGED_CODE" or not entry.get("cleaned") or entry.get("already_clean"):
            continue
        code = str(entry.get("generated_code", "") or "").strip()
        if not code or "def clean_column" not in code:
            continue

        lines = code.splitlines()
        entry_imports = [ln for ln in lines if ln.strip().startswith(("import ", "from "))]
        body = "\n".join(ln for ln in lines if ln not in entry_imports).strip()
        if not body:
            continue
        for line in entry_imports:
            if line not in imports:
                imports.append(line)

        name = f"clean_{_safe_name(str(key))}"
        functions.append(f"# {key}\n" + re.sub(r"\bdef\s+clean_column\s*\(", f"def {name}(", body, count=1))
        if entry.get("target_columns"):
            # A dependency task is handed the whole table, because it changes more than one column.
            calls.append(f"    df = {name}(df)")
        else:
            calls.append(f"    df[{str(key)!r}] = {name}(df[{str(key)!r}])")

    if functions and "import pandas as pd" not in imports:
        imports.insert(0, "import pandas as pd")
    return imports, functions, calls


def merged_cleaning_code(report: Any) -> str:
    """One module's worth of code: the imports, the task functions and `clean_dataset`."""
    imports, functions, calls = cleaning_functions(report)
    if not functions:
        return ""
    wrapper = "\n".join(
        ["def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:", "    df = df.copy()", *calls, "    return df"]
    )
    return "\n".join(imports).strip() + "\n\n" + "\n\n".join(functions) + "\n\n" + wrapper


def _markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build_notebook(report: Any, dataset_path: str, output_path: str = "cleaned.csv") -> dict:
    """A notebook that reproduces this run: load, clean, save.

    `dataset_path` is the dirty file the run was given, and `output_path` is where the notebook
    writes its cleaned copy. Both are written into the notebook as plain strings, so whoever opens
    it can point them somewhere else.
    """
    imports, functions, calls = cleaning_functions(report)
    data = _as_report_dict(report)
    runtime = data.get("runtime_seconds")
    tokens = (data.get("total_usage") or {}).get("total_tokens")

    made_with = "This notebook was written by MADClean from one cleaning run."
    if runtime:
        made_with += f" That run took {float(runtime):.1f} seconds"
        made_with += f" and spent {int(tokens):,} tokens." if tokens else "."

    cells: list[dict] = [
        _markdown(
            "# Cleaning this dataset\n"
            "\n"
            f"{made_with} The agents' work ends as the code below: one function per column, and "
            "one per dependency rule between columns. Running this notebook from top to bottom "
            "reproduces the cleaned table without calling a model, so it costs nothing and gives "
            "the same answer every time.\n"
        ),
        _code(
            "import pandas as pd\n"
            "\n"
            f"dirty_path = {dataset_path!r}\n"
            f"cleaned_path = {output_path!r}\n"
            "\n"
            "df = pd.read_csv(dirty_path)\n"
            "print(f'{len(df):,} rows, {len(df.columns)} columns')\n"
            "df.head()"
        ),
    ]

    if functions:
        other_imports = [line for line in imports if line != "import pandas as pd"]
        if other_imports:
            cells.append(_code("\n".join(other_imports)))
        cells.append(
            _markdown(
                "## What the agents wrote\n"
                "\n"
                "One function per task, exactly as the coding agent wrote it and the run applied "
                "it.\n"
            )
        )
        cells.extend(_code(function) for function in functions)
        cells.append(
            _markdown(
                "## Applying them\n"
                "\n"
                "Columns first, then the dependency rules, which is the order the run used: a "
                "rule between two columns only makes sense once both have been cleaned.\n"
            )
        )
        cells.append(
            _code(
                "def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:\n"
                "    df = df.copy()\n"
                + "\n".join(calls)
                + "\n    return df\n"
                "\n"
                "cleaned = clean_dataset(df)\n"
                "cleaned.head()"
            )
        )
    else:
        cells.append(
            _markdown(
                "## Nothing to apply\n"
                "\nThis run changed no column, so there is no cleaning code to reproduce.\n"
            )
        )
        cells.append(_code("cleaned = df.copy()"))

    cells.append(
        _code(
            "changed = (df.astype(str) != cleaned.astype(str)).sum().sum()\n"
            "print(f'{changed:,} cells differ from the dirty file')\n"
            "cleaned.to_csv(cleaned_path, index=False)\n"
            "print(f'written to {cleaned_path}')"
        )
    )

    return {"cells": cells, "metadata": KERNEL, **NOTEBOOK_FORMAT}


def notebook_json(report: Any, dataset_path: str, output_path: str = "cleaned.csv") -> str:
    """The notebook as the text of an `.ipynb` file."""
    return json.dumps(build_notebook(report, dataset_path, output_path), indent=1) + "\n"
