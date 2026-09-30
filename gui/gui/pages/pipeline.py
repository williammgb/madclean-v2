"""The Pipeline view: one row per cleaning task, and an inspector for the step you pick.

A task is a column, or a dependency rule between columns. Its row shows the agents' steps in the
order they happened — recommender, coder, validator, and another attempt whenever the validator
sent the work back — then how the task ended. The inspector shows one step at a time; its arrows
walk through the steps of the same task.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from ..components.icons import icon
from ..components.shell import view, view_head
from ..state import State


def tone_chip(text, tone: rx.Var) -> rx.Component:
    """A pill whose colour comes from the state: green, red, orange, accent, or none."""
    return rx.el.span(
        text,
        class_name=rx.cond(tone.to(str) == "", "chip", "chip chip-" + tone.to(str)),
    )


def step_mark(step: rx.Var) -> rx.Component:
    """The tick, cross or dash in front of a step's name; a pulse while it waits for you."""
    mark = step["mark"].to(str)
    return rx.cond(
        step["needs_blink"] == "1",
        rx.el.span(class_name="pulse"),
        rx.cond(
            mark == "check",
            icon("check"),
            rx.cond(mark == "x", icon("x"), rx.cond(mark == "minus", icon("minus"), rx.fragment())),
        ),
    )


def step_chip(column: rx.Var, step: rx.Var, index: rx.Var) -> rx.Component:
    """One step of one task, coloured by how it ended, with an arrow from the step before."""
    picked = (State.pipeline_inspector["column"].to(str) == column.to(str)) & (
        State.pipeline_inspector["step"].to(str) == step["id"].to(str)
    )
    return rx.fragment(
        rx.cond(index > 0, rx.el.span(icon("right"), class_name="step-sep"), rx.fragment()),
        rx.el.button(
            step_mark(step),
            step["label"],
            type="button",
            class_name="step " + step["chip"].to(str),
            aria_pressed=rx.cond(picked, "true", "false"),
            on_click=State.open_pipeline_step(column, step["id"]),
        ),
    )


def task_row(row: rx.Var) -> rx.Component:
    column = row["column"]
    return rx.el.div(
        rx.el.span(
            rx.cond(
                row["type_color"].to(str) != "",
                rx.el.span(class_name="type-dot", style={"--c": row["type_color"].to(str)}),
                rx.fragment(),
            ),
            column,
            class_name="nm",
            title=column.to(str),
        ),
        rx.el.div(
            rx.foreach(
                row["steps"].to(list[dict[str, str]]),
                lambda step, i: step_chip(column, step, i),
            ),
            class_name="steps",
        ),
        rx.el.span(tone_chip(row["result"], row["result_tone"]), class_name="result"),
        class_name="task",
    )


def inspector() -> rx.Component:
    """One step: what it is, how it ended, and what it produced."""
    step = State.pipeline_inspector
    return rx.el.aside(
        rx.el.div(
            rx.el.h2(step["title"].to(str)),
            tone_chip(step["status"].to(str), step["tone"]),
            rx.el.button(
                icon("left"),
                type="button",
                class_name="btn btn-ghost btn-icon btn-sm",
                aria_label="Previous step",
                disabled=~step["has_prev"].to(bool),
                on_click=State.move_pipeline_step(-1),
            ),
            rx.el.button(
                icon("right"),
                type="button",
                class_name="btn btn-ghost btn-icon btn-sm",
                aria_label="Next step",
                disabled=~step["has_next"].to(bool),
                on_click=State.move_pipeline_step(1),
            ),
            class_name="insp-head",
        ),
        rx.el.div(
            rx.el.pre(
                rx.foreach(step["lines"].to(list[str]), lambda line: rx.el.div(line)),
                class_name="code",
            ),
            class_name="insp-body",
        ),
        class_name="panel inspector",
        aria_live="polite",
    )


def result_chips() -> rx.Component:
    counts = State.pipeline_result_counts
    failed = counts["failed"].to(int)
    return rx.el.div(
        rx.el.span(counts["validated"].to(str) + " validated", class_name="chip chip-green"),
        rx.el.span(counts["cleaned"].to(str) + " cleaned", class_name="chip chip-accent"),
        rx.el.span(counts["clean"].to(str) + " already clean", class_name="chip"),
        rx.el.span(
            failed.to(str) + " failed", class_name=rx.cond(failed > 0, "chip chip-red", "chip")
        ),
        class_name="summary-row",
    )


def page() -> rx.Component:
    has_rows = State.pipeline_flow_rows.length() > 0
    return view(
        "pipeline",
        view_head("Pipeline", rx.cond(has_rows, result_chips(), rx.fragment())),
        rx.cond(
            has_rows,
            rx.el.div(
                rx.el.div(
                    rx.foreach(State.pipeline_flow_rows.to(list[dict[str, Any]]), task_row),
                    class_name="panel",
                ),
                inspector(),
                class_name="pipe-layout",
            ),
            rx.el.div(rx.el.h2("No run yet"), class_name="panel empty"),
        ),
    )
