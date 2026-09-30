"""Every view builds, and the GUI is made of page modules rather than one file.

Reflex turns a page into components when the app compiles; a view that refers to a state field
that does not exist, or mixes types it cannot render, fails there. Building each view here catches
that in the fast gate instead of when the app is started.
"""

import sys
from pathlib import Path

import pytest
import reflex as rx

GUI_DIR = Path(__file__).resolve().parents[1] / "gui"
sys.path.insert(0, str(GUI_DIR))

from gui.components.shell import rail, top_bar  # noqa: E402
from gui.pages import (  # noqa: E402
    evaluation,
    guide,
    logs,
    pipeline,
    profile,
    report,
    review,
    settings,
    table,
)

VIEWS = {
    "table": table,
    "profile": profile,
    "pipeline": pipeline,
    "review": review,
    "logs": logs,
    "report": report,
    "evaluation": evaluation,
    "guide": guide,
    # Not in the rail, but a view like any other: the gear in the top bar opens it.
    "settings": settings,
}


@pytest.mark.parametrize("name", sorted(VIEWS))
def test_every_view_builds(name):
    component = VIEWS[name].page()
    assert isinstance(component, rx.Component)
    # A view that rendered nothing at all would pass every other check here.
    assert str(component).strip() != ""


def test_the_shell_builds():
    assert isinstance(top_bar(), rx.Component)
    assert isinstance(rail(), rx.Component)


def test_the_rail_offers_the_nine_views():
    from gui.components.shell import NAV_GROUPS

    names = [name for _label, items in NAV_GROUPS for name, _title, _icon in items]
    assert names == [
        "table", "profile", "pipeline", "review", "logs", "report", "evaluation", "settings", "guide"
    ]


def test_the_gui_is_split_into_modules_rather_than_one_file():
    """The page file assembles; the views and the shell live beside it."""
    page_file = GUI_DIR / "gui" / "gui.py"
    assert page_file.read_text(encoding="utf-8").count("\n") < 80, "gui.py is assembling, not building"

    for module in VIEWS:
        assert (GUI_DIR / "gui" / "pages" / f"{module}.py").is_file()
    # Every view the shell can open has a module behind it, so no link lands on a blank page.
    from gui.components.shell import NAV_GROUPS

    for _label, items in NAV_GROUPS:
        for name, _title, _icon in items:
            assert name in VIEWS
    assert (GUI_DIR / "gui" / "components" / "shell.py").is_file()
    assert (GUI_DIR / "gui" / "components" / "icons.py").is_file()


def _view_source() -> str:
    parts = [(GUI_DIR / "gui" / "pages" / f"{module}.py").read_text(encoding="utf-8") for module in VIEWS]
    parts.append((GUI_DIR / "gui" / "components" / "shell.py").read_text(encoding="utf-8"))
    return "".join(parts)


def test_every_event_handler_is_reachable():
    """A handler no view calls is a feature the rebuild dropped, or code nothing runs.

    The rebuild once left the column-instructions menu and the "Label cells" switch opening
    dialogs that no view drew; this is the check that would have caught it.
    """
    import re

    from gui.state import State

    views = _view_source()
    state_source = (GUI_DIR / "gui" / "state.py").read_text(encoding="utf-8")
    # Only the handlers written in state.py: Reflex adds a setter for every field on its own.
    written = set(re.findall(r"^    (?:async )?def ([a-z]\w*)\(self", state_source, re.MULTILINE))
    unreached = sorted(
        name
        for name in written & set(State.event_handlers)
        if f"State.{name}" not in views
        and f"State.{name}" not in state_source
        and f"self.{name}(" not in state_source
    )
    assert unreached == []


def test_the_changed_cell_count_covers_the_whole_table():
    """The report's "Cells changed" tile once read a list nothing filled, so it always said 0."""
    import pandas as pd

    from gui.state import State

    original = pd.DataFrame({"a": [1, 2, None, 4], "b": ["x", "y", "z", None]})
    cleaned = pd.DataFrame({"a": [1.0, 3.0, None, 4.0], "b": ["x", "Y", "z", "w"]})
    # 1 vs 1.0 is no change and both-missing is no change: a[1], b[1] and b[3] changed.
    assert State._count_changed_cells(original, cleaned) == 3
    assert State._count_changed_cells(None, cleaned) == 0


def test_evaluation_splits_found_cells_into_repaired_and_still_wrong():
    """A wrong cell changed to another wrong value is not "Repaired"; it gets its own segment."""
    import pandas as pd

    from gui.state import evaluation_view
    from madclean.evaluation import compute_cleaning_metrics

    dirty = pd.DataFrame({"a": ["1O", "x", "3"], "b": ["p", "q", "r"]})
    truth = pd.DataFrame({"a": ["10", "2", "3"], "b": ["p", "q", "r"]})
    cleaned = pd.DataFrame({"a": ["10", "y", "3"], "b": ["p", "q", "r"]})
    shown = evaluation_view(compute_cleaning_metrics(dirty, cleaned, truth), ["a", "b"], {})

    wrong, changed = shown["bars"]
    assert [(s["label"], s["count"]) for s in wrong["segments"]] == [
        ("Repaired", "1"), ("Changed, still wrong", "1"), ("Left wrong", "0")
    ]
    assert changed["total"] == "2"
    assert [s["label"] for s in changed["segments"]] == ["Repaired", "Changed, still wrong", "Changed but was fine"]
    assert [s["v"] for s in shown["scores"]] == ["50.00", "50.00", "50.00"]

    row_a, row_b = shown["rows"]
    assert (row_a["repaired"], row_a["repaired_class"]) == ("1", "num")
    assert (row_a["fp"], row_a["fp_class"]) == ("0", "num score-na")
    # Nothing to repair in b: its scores are undefined, shown as a dash, and it has no F1 bar.
    assert (row_b["precision"], row_b["has_f1"]) == ("—", "0")

    clean_run = evaluation_view(compute_cleaning_metrics(dirty, truth, truth), ["a", "b"], {})
    assert "Changed, still wrong" not in [s["label"] for s in clean_run["bars"][0]["segments"]]


def test_the_report_shows_each_agents_tokens():
    """The panel once read "label" and "pct" from rows that had neither, so it showed "undefined %"."""
    import re

    from gui.state import token_usage_rows

    usage = {
        "recommender": {"input_tokens": 50, "output_tokens": 25, "total_tokens": 75},
        "coding": {"input_tokens": 20, "output_tokens": 5},
        "validation": {"total_tokens": 0},
        "total_usage": {"total_tokens": 100},
    }
    rows = token_usage_rows(usage)
    assert [(r["label"], r["pct"]) for r in rows] == [("Recommender · 75", "75.0"), ("Coder · 25", "25.0")]
    assert all(r["color"] for r in rows)
    assert token_usage_rows({}) == [] and token_usage_rows(None) == []

    report_source = (GUI_DIR / "gui" / "pages" / "report.py").read_text(encoding="utf-8")
    panel = report_source.split("def token_split", 1)[1].split("\ndef ", 1)[0]
    assert set(re.findall(r'row\["(\w+)"\]', panel)) == set(rows[0])


def test_the_validator_fallback_choices_are_ones_the_config_accepts():
    """The dropdown once sent "accept", which the setter silently refused."""
    import re

    settings_source = (GUI_DIR / "gui" / "pages" / "settings.py").read_text(encoding="utf-8")
    block = settings_source.split("When undecided", 1)[1].split("value=State", 1)[0]
    offered = set(re.findall(r'value="([a-z_]+)"', block))
    assert offered == {"accept_cleaned", "leave_uncleaned", "ask_user"}


def test_the_views_carry_the_approved_previews_class_names():
    """The stylesheet is the preview's, so the markup has to use its names to be styled by it."""
    stylesheet = (GUI_DIR / "assets" / "madclean.css").read_text(encoding="utf-8")
    for class_name in ("topbar", "nav-item", "view-head", "panel", "grid", "steps-list", "tiles"):
        assert f".{class_name}" in stylesheet

    markup = "".join(
        (GUI_DIR / "gui" / "pages" / f"{module}.py").read_text(encoding="utf-8") for module in VIEWS
    )
    markup += (GUI_DIR / "gui" / "components" / "shell.py").read_text(encoding="utf-8")
    for class_name in ("topbar", "nav-item", "view-head", "panel", "grid"):
        assert f'"{class_name}' in markup or f" {class_name}" in markup
