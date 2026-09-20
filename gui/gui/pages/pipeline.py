"""The Pipeline view: one row per cleaning task, and an inspector for the step you pick.

A task is a column, or a dependency rule between columns. Its row shows the agents' steps in the
order they happened — recommender, coder, validator, and another attempt whenever the validator
sent the work back. Clicking a step opens what that step actually produced.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from ..components.icons import icon
from ..components.shell import chip, panel, panel_head, view, view_head
from ..state import State


def step_chip(column: rx.Var, step: rx.Var) -> rx.Component:
    """One step of one task, coloured by how it ended."""
    status_class = rx.cond(
        step["needs_blink"] == "1",
        "step waiting",
        rx.cond(
            step["is_error"] == "1",
            "step rejected",
            rx.cond(step["is_success"] == "1", "step ok", "step running"),
        ),
    )
    return rx.fragment(
        rx.el.button(
            rx.cond(step["needs_blink"] == "1", rx.el.span(class_name="pulse"), rx.fragment()),
            step["title"],
            type="button",
            class_name=rx.cond(
                step["is_active"] == "1", status_class + " is-active", status_class
            ),
            on_click=State.select_pipeline_flow_step(column, step["id"]),
        ),
        rx.cond(
            step["show_arrow_after"] == "1",
            rx.el.span(icon("arrow"), class_name="step-sep"),
            rx.fragment(),
        ),
    )


def task_row(row: rx.Var) -> rx.Component:
    column = row["column"]
    return rx.el.div(
        rx.el.div(
            rx.el.span(column, class_name="nm"),
            rx.el.span(
                rx.cond(column.to(str).contains("→"), "dependency rule", "column"),
                class_name="small faint",
            ),
            class_name="who",
        ),
        rx.el.div(
            rx.foreach(
                row["steps"].to(list[dict[str, str]]),
                lambda step: step_chip(column, step),
            ),
            class_name="steps",
        ),
        rx.el.div(
            rx.cond(
                row["column_done"] == "1",
                chip("done", tone="green", large=False),
                chip("running", tone="accent", large=False),
            ),
            class_name="result",
        ),
        class_name="task",
    )


def inspector() -> rx.Component:
    """What the selected step produced, for whichever task has a step open."""
    return rx.el.aside(
        panel_head(rx.el.h2("Step detail")),
        rx.el.div(
            rx.foreach(
                State.opened_pipeline_rows.to(list[dict[str, Any]]),
                lambda row: rx.el.ul(
                    rx.el.li(
                        rx.el.div(
                            rx.el.span(
                                row["column"].to(str) + " · " + row["active_title"].to(str),
                                class_name="t",
                            ),
                            rx.el.span(row["active_status"], class_name="small faint"),
                            class_name="tl-head",
                        ),
                        rx.el.pre(
                            rx.foreach(
                                row["active_output_lines"].to(list[str]),
                                lambda line: rx.el.div(line),
                            ),
                            class_name="code",
                        ),
                    ),
                    class_name="tl",
                ),
            ),
            rx.cond(
                State.opened_pipeline_rows.length() == 0,
                rx.el.p(
                    "Pick a step in a task to see what it produced.",
                    class_name="muted small",
                ),
                rx.fragment(),
            ),
            class_name="insp-body",
        ),
        class_name="panel inspector",
        aria_live="polite",
    )


def page() -> rx.Component:
    return view(
        "pipeline",
        view_head(
            "Pipeline",
            "Columns are cleaned first and at the same time; a dependency rule waits until both "
            "of its columns are done.",
            rx.el.div(
                rx.cond(
                    State.pipeline_task_count > 0,
                    chip(State.pipeline_task_count.to(str) + " tasks"),
                    rx.fragment(),
                ),
                rx.cond(
                    State.review_waiting_count > 0,
                    chip(State.review_waiting_count.to(str) + " waiting for you", tone="orange"),
                    rx.fragment(),
                ),
                class_name="summary-row",
            ),
        ),
        rx.cond(
            State.pipeline_flow_rows.length() > 0,
            rx.el.div(
                rx.el.div(
                    rx.foreach(State.pipeline_flow_rows.to(list[dict[str, Any]]), task_row),
                    class_name="panel",
                ),
                inspector(),
                class_name="pipe-layout",
            ),
            rx.el.div(
                rx.el.p(State.selected_trace_status, class_name="muted"),
                class_name="panel panel-body",
            ),
        ),
    )
