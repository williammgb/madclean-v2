"""The Logs view: what the run printed, newest at the bottom."""

from __future__ import annotations

import reflex as rx

from ..components.shell import panel, panel_head, view, view_head
from ..state import State


def page() -> rx.Component:
    return view(
        "logs",
        view_head(
            "Logs",
            "Everything the run printed, as it printed it. The same lines go to the log file the "
            "run writes beside the dataset.",
        ),
        panel(
            panel_head(
                rx.el.h2("Run log"),
                rx.el.span(State.status_msg, class_name="small muted"),
            ),
            rx.cond(
                State.logs.length() > 0,
                rx.el.pre(
                    rx.foreach(State.logs, lambda line: rx.el.div(rx.el.span(line))),
                    class_name="console",
                ),
                rx.el.div(
                    rx.el.p("Nothing logged yet. Start a run to see it here.", class_name="muted"),
                    class_name="panel-body",
                ),
            ),
        ),
    )
