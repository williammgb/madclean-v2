"""The MADClean page: the shell from the approved preview, and one module per view.

Everything visual lives in `gui/assets/madclean.css`, which is the preview's own stylesheet, so
the components here carry its class names and add no styling of their own.
"""

import reflex as rx

from .components.icons import sprite
from .components.shell import rail, top_bar
from .pages import (
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
from .state import State


def main() -> rx.Component:
    """The nine views, in the order the rail lists them. One is shown at a time."""
    return rx.el.main(
        table.page(),
        profile.page(),
        pipeline.page(),
        review.page(),
        logs.page(),
        report.page(),
        evaluation.page(),
        settings.page(),
        guide.page(),
        class_name="main",
    )


@rx.page(title="MADClean")
def index() -> rx.Component:
    return rx.fragment(
        sprite(),
        rx.el.div(
            top_bar(),
            rail(),
            main(),
            class_name=rx.cond(State.rail_collapsed == "1", "app collapsed", "app"),
        ),
    )


app = rx.App(
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap",
        "/madclean.css",
    ],
)
