"""The Logs view: what the run printed, newest at the bottom, one line per row."""

from __future__ import annotations

import reflex as rx

from ..components.icons import icon
from ..components.shell import panel, panel_head, view, view_head
from ..state import State


def log_line(row: rx.Var) -> rx.Component:
    """One line: its task tag in the accent colour, then the text in the colour of its tone."""
    return rx.el.div(
        rx.el.span(
            rx.cond(
                row["col"] != "",
                rx.fragment(rx.el.span(row["col"], class_name="col"), " "),
                rx.fragment(),
            ),
            rx.el.span(row["text"], class_name=row["tone"].to(str)),
        ),
        class_name="ln",
    )


def page() -> rx.Component:
    return view(
        "logs",
        view_head("Logs"),
        panel(
            panel_head(
                rx.el.div(
                    rx.el.h2("Run log"),
                    rx.el.span(State.run_chip_text, class_name=State.run_chip_class),
                    class_name="log-tools",
                ),
                rx.el.div(
                    rx.el.span(State.logs.length().to(str) + " lines", class_name="small faint"),
                    rx.el.button(
                        icon("copy"),
                        "Copy",
                        type="button",
                        class_name="btn btn-ghost btn-sm",
                        disabled=State.logs.length() == 0,
                        on_click=State.copy_logs,
                    ),
                    class_name="log-tools",
                ),
            ),
            rx.cond(
                State.logs.length() > 0,
                rx.el.pre(rx.foreach(State.log_rows, log_line), class_name="console"),
                rx.el.div(rx.el.h2("No log yet"), class_name="empty"),
            ),
        ),
    )
