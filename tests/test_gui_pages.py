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


def test_the_rail_offers_the_eight_views():
    from gui.components.shell import NAV_GROUPS

    names = [name for _label, items in NAV_GROUPS for name, _title, _icon in items]
    assert names == ["table", "profile", "pipeline", "review", "logs", "report", "evaluation", "guide"]


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
