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


def test_a_repaired_column_shows_validator_repair_validator_with_a_violet_repair():
    from gui.state import State

    steps = [
        ("validator_1", "Validator (attempt 1)", "needs_correction"),
        ("repair_1", "Repair (attempt 1)", "ok"),
        ("validator_repair_1", "Validator (after repair)", "approved"),
    ]

    chips = [State._step_chip(step_id, title, status, False) for step_id, title, status in steps]

    # Validator ✕ › Repair ✓ › Validator ✓
    assert [(chip["label"], chip["mark"]) for chip in chips] == [
        ("Validator", "x"), ("Repair", "check"), ("Validator", "check"),
    ]
    assert [chip["chip"] for chip in chips] == ["rejected", "repair", "ok"]
    # A repair in a later round still reads Repair: it belongs to the validator before it.
    assert State._step_chip("repair_3", "Repair (attempt 3)", "ok", False)["label"] == "Repair"
    css = (GUI_DIR / "assets" / "madclean.css").read_text(encoding="utf-8")
    assert ".step.repair{background:var(--violet-soft);color:var(--violet-text)}" in css
    # The light theme and both dark themes each define the violet tokens.
    assert css.count("--violet-soft:") == 3 and css.count("--violet-text:") == 3


def _open_call(agent, task, seconds, attempt=1, try_no=1, tries=None, reason=None):
    """One open call as `LLMActivity.snapshot()` lists it."""
    return {
        "agent": agent, "task": task, "attempt": attempt, "model": "gemini-2.5-pro",
        "seconds": seconds, "try_no": try_no, "tries": tries, "reason": reason,
    }


def test_the_api_line_says_which_model_how_many_calls_and_how_long():
    from gui.state import api_activity_view

    two = api_activity_view({
        "open": [_open_call("Recommender", "abv", 31.2), _open_call("Validator", "city", 48.6, attempt=2)],
        "last_reply_seconds": 3.0,
    })
    assert two["api_line_text"] == "gemini-2.5-pro · 2 calls · 0:48"
    assert (two["api_line_bold"], two["api_line_rest"]) == ("gemini-2.5-pro", " · 2 calls · 0:48")
    assert two["api_line_class"] == "api"
    # The hover lists every open call, oldest first.
    assert two["api_line_title"] == "Validator · city · 0:48\nRecommender · abv · 0:31"
    assert two["api_open_by_task"] == {
        "city": {"label": "Validator 2", "secs": "0:48", "chip": "running"},
        "abv": {"label": "Recommender", "secs": "0:31", "chip": "running"},
    }

    slow = api_activity_view({"open": [_open_call("Coder", "pages", 134.0)], "last_reply_seconds": 1.0})
    assert slow["api_line_text"] == "gemini-2.5-pro · 1 call · 2:14"
    assert slow["api_line_class"] == "api slow"
    assert slow["api_open_by_task"]["pages"]["chip"] == "slow"
    # Two minutes on the dot is not slow yet.
    assert api_activity_view({"open": [_open_call("Coder", "pages", 120.0)]})["api_line_class"] == "api"

    local = api_activity_view({"open": [], "last_reply_seconds": 3.7})
    assert local["api_line_text"] == "Working locally · last reply 0:03"
    assert local["api_line_class"] == "api local" and local["api_open_by_task"] == {}
    before_first = api_activity_view({"open": [], "last_reply_seconds": None})
    assert before_first["api_line_text"] == "Working locally"

    retry = api_activity_view({
        "open": [
            _open_call("Recommender", "abv", 50.0),
            _open_call("Coder", "pages", 9.0, try_no=2, tries=3, reason="timed out"),
        ],
        "last_reply_seconds": 2.0,
    })
    assert retry["api_line_text"] == "gemini-2.5-pro · timed out · try 2 of 3"
    assert retry["api_line_class"] == "api retry"
    assert retry["api_line_title"].splitlines()[1] == "Coder · pages · try 2 of 3 · 0:09"
    assert retry["api_open_by_task"]["pages"] == {"label": "Coder · try 2 of 3", "secs": "0:09", "chip": "slow"}


def test_an_open_call_ends_its_task_row_with_a_pulsing_chip():
    from gui.state import State

    chip = State._open_call_chip({"label": "Recommender", "secs": "0:31", "chip": "running"})
    assert (chip["label"], chip["secs"], chip["chip"], chip["mark"]) == ("Recommender", "0:31", "running", "pulse")
    assert State._open_call_chip({"label": "Coder", "secs": "2:14", "chip": "slow"})["chip"] == "slow"
    # Every other chip has no clock, so the row draws none.
    assert State._step_chip("coder_1", "Coder (attempt 1)", "ok", False)["secs"] == ""
    css = (GUI_DIR / "assets" / "madclean.css").read_text(encoding="utf-8")
    assert ".step.slow{" in css and ".api.slow" in css and ".api .ring{" in css
    assert "@media (prefers-reduced-motion:reduce){.pulse{animation:none}}" in css


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
