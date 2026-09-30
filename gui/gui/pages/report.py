"""The Report view: what the run cost, and the code it wrote.

The tiles are the run at a glance. Below them is the cleaning code the coder produced, per column
and per dependency rule, which is the same code the notebook export writes out.
"""

from __future__ import annotations

import reflex as rx

from ..components.icons import icon
from ..components.shell import panel, panel_head, view, view_head
from ..state import State


def tile(label: str, value, note: str = "") -> rx.Component:
    return rx.el.div(
        rx.el.span(label, class_name="k"),
        rx.el.span(value, class_name="v"),
        rx.cond(note != "", rx.el.span(note, class_name="s"), rx.fragment()),
        class_name="panel tile",
    )


def token_split() -> rx.Component:
    """Tokens per agent, as the preview's row of proportions."""
    return panel(
        panel_head(rx.el.h2("Tokens per agent")),
        rx.el.div(
            rx.foreach(
                State.token_usage_rows.to(list[dict[str, str]]),
                lambda row: rx.el.div(
                    rx.el.span(row["label"], class_name="lbl"),
                    rx.el.span(
                        rx.el.span(
                            style={"width": row["pct"].to(str) + "%", "background": row["color"]}
                        ),
                        class_name="track",
                    ),
                    rx.el.span(row["pct"].to(str) + "%", class_name="n"),
                    class_name="bar-row",
                ),
            ),
            class_name="bars panel-body",
        ),
    )


def generated_code() -> rx.Component:
    return panel(
        panel_head(
            rx.el.h2("Cleaning code"),
            rx.el.select(
                rx.foreach(State.report_code_keys, lambda key: rx.el.option(key, value=key)),
                value=State.selected_report_code_key,
                on_change=State.set_selected_report_code_key,
                class_name="select",
                style={"width": "260px"},
            ),
        ),
        rx.el.div(
            rx.el.div(
                rx.foreach(
                    State.selected_report_meta_lines, lambda line: rx.el.span(line, class_name="chip")
                ),
                class_name="summary-row",
            ),
            rx.el.pre(
                rx.foreach(
                    State.generated_code_lines_for_selected, lambda line: rx.el.div(line)
                ),
                class_name="code",
            ),
            class_name="panel-body",
            style={"display": "flex", "flexDirection": "column", "gap": "12px"},
        ),
    )


def exports() -> rx.Component:
    """Everything the run can hand you to take away."""
    return panel(
        panel_head(rx.el.h2("Export")),
        rx.el.div(
            rx.el.button(
                icon("table"),
                "Cleaned data (.csv)",
                type="button",
                class_name="btn btn-soft",
                on_click=State.download_cleaned_file,
            ),
            rx.el.button(
                icon("report"),
                "Cleaning code (.py)",
                type="button",
                class_name="btn btn-soft",
                on_click=State.download_cleaning_code,
            ),
            rx.el.button(
                icon("report"),
                "Notebook (.ipynb)",
                type="button",
                class_name="btn btn-soft",
                on_click=State.download_notebook,
            ),
            class_name="panel-body summary-row",
        ),
    )


def page() -> rx.Component:
    return view(
        "report",
        view_head("Report"),
        rx.cond(
            State.has_cleaned,
            rx.fragment(
                rx.el.div(
                    tile("Runtime", State.runtime_seconds_display + " s"),
                    tile("Tokens", State.total_tokens_display),
                    tile("Cells changed", State.changed_cell_count.to(str)),
                    tile("Tasks", State.pipeline_task_count.to(str)),
                    class_name="tiles",
                ),
                token_split(),
                exports(),
                generated_code(),
            ),
            rx.el.div(rx.el.h2("No run yet"), class_name="panel empty"),
        ),
    )
