"""The Table view: the data itself, with each column's semantic type in its header band.

Rows come from the state already prepared: every row carries the values to show, a flag per column
saying whether the cleaning changed that cell, and the value it had before. A changed cell shows
both, the old struck through above the new, which is what the preview does.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from ..components.icons import icon
from ..components.shell import view, view_head
from ..state import State


def column_header(col: rx.Var, index: rx.Var) -> rx.Component:
    """One header: the colour band on top, the name, and the semantic type underneath."""
    colour = State.display_column_header_colors[col]
    return rx.el.th(
        rx.el.div(
            rx.el.div(
                rx.el.span(col, class_name="th-name"),
                rx.el.button(
                    icon("more"),
                    type="button",
                    class_name="th-menu",
                    aria_label=f"Column menu",
                    on_click=State.open_recommender_hint_editor(col),
                ),
                class_name="th-top",
            ),
            rx.el.div(
                rx.el.span(class_name="type-dot"),
                State.column_semantic_types[col],
                rx.cond(
                    State.column_fd_markers[col] != "",
                    rx.el.span(State.column_fd_markers[col], class_name="fd-badge"),
                    rx.fragment(),
                ),
                class_name="th-type",
            ),
            class_name="th",
        ),
        style={"--c": colour},
    )


def cell(row: rx.Var, col: rx.Var, index: rx.Var) -> rx.Component:
    """One cell. Changed cells show what the value was, and what it became."""
    value = row[col].to(str)
    changed = row["__modified_flags"].to(list[bool])[index]
    before = row["__original_values"].to(list[str])[index]
    return rx.el.td(
        rx.cond(
            changed,
            rx.fragment(
                rx.el.span(rx.cond(before == "", "(empty)", before), class_name="old"),
                rx.el.span(rx.cond(value == "", "(empty)", value), class_name="new"),
            ),
            rx.cond(
                (value == "") | (value == "nan") | (value == "None"),
                rx.el.span("(empty)", class_name="null"),
                rx.el.span(value),
            ),
        ),
        class_name=rx.cond(changed, "chg", ""),
    )


def grid() -> rx.Component:
    return rx.el.div(
        rx.el.table(
            rx.el.thead(
                rx.el.tr(
                    rx.el.th("#", class_name="rownum"),
                    rx.foreach(State.column_names, column_header),
                )
            ),
            rx.el.tbody(
                rx.foreach(
                    State.df_preview_window.to(list[dict[str, Any]]),
                    lambda row: rx.el.tr(
                        rx.el.td(row["row_id"], class_name="rownum"),
                        rx.foreach(State.column_names, lambda col, i: cell(row, col, i)),
                    ),
                )
            ),
            class_name="grid",
        ),
        class_name="table-wrap",
    )


def toolbar() -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.el.button(
                "Original",
                type="button",
                aria_pressed=rx.cond(State.has_cleaned, "false", "true"),
                disabled=True,
                title="The uploaded data",
            ),
            rx.el.button(
                "Cleaned",
                type="button",
                aria_pressed=rx.cond(State.has_cleaned, "true", "false"),
                disabled=~State.has_cleaned,
                title="The data after the run",
            ),
            class_name="seg",
            role="group",
            aria_label="Table version",
        ),
        rx.el.div(
            rx.el.label(
                rx.el.input(
                    type="checkbox",
                    checked=State.show_fds,
                    on_change=State.set_show_fds,
                ),
                rx.el.span(class_name="track"),
                "Dependencies",
                class_name="switch",
            ),
            rx.el.label(
                rx.el.input(
                    type="checkbox",
                    checked=State.label_cells_mode,
                    on_change=State.set_label_cells_mode,
                ),
                rx.el.span(class_name="track"),
                "Label cells",
                class_name="switch",
            ),
            class_name="toggles",
        ),
        rx.el.span(class_name="spacer"),
        rx.el.div(
            State.page_range_label,
            rx.el.button(
                icon("left"),
                type="button",
                class_name="btn btn-ghost btn-icon btn-sm",
                aria_label="Previous page",
                disabled=State.page_offset <= 0,
                on_click=State.prev_page,
            ),
            rx.el.button(
                icon("right"),
                type="button",
                class_name="btn btn-ghost btn-icon btn-sm",
                aria_label="Next page",
                disabled=(State.page_offset + State.page_size) >= State.total_rows,
                on_click=State.next_page,
            ),
            class_name="pager",
        ),
        class_name="toolbar",
    )


def dependency_strip() -> rx.Component:
    """The dependencies the profiler found, as the preview's row of chips."""
    return rx.cond(
        State.show_fds & (State.fd_button_rows.length() > 0),
        rx.el.div(
            rx.el.span("Dependencies found:", class_name="small muted"),
            rx.foreach(
                State.fd_button_rows,
                lambda row: rx.el.button(
                    row["label"],
                    rx.el.span(row["score"], class_name="score"),
                    type="button",
                    class_name=rx.cond(row["active"] == "1", "fd-chip is-on", "fd-chip"),
                    aria_pressed=rx.cond(row["active"] == "1", "true", "false"),
                    on_click=State.toggle_fd_selection(row["lhs"], row["rhs"]),
                ),
            ),
            class_name="fd-strip",
        ),
        rx.fragment(),
    )


def page() -> rx.Component:
    return view(
        "table",
        view_head(
            "Table",
            "Header colours show each column's semantic type from profiling. A changed cell shows "
            "what it was above what it became.",
        ),
        rx.cond(
            State.column_names.length() > 0,
            rx.fragment(
                toolbar(),
                dependency_strip(),
                grid(),
                rx.el.div(
                    rx.el.span(State.table_window_label),
                    rx.cond(
                        State.has_cleaned,
                        rx.el.span(State.changed_cells_label),
                        rx.fragment(),
                    ),
                    class_name="table-foot",
                ),
            ),
            rx.el.div(
                rx.el.p("Upload a CSV or Excel file to begin.", class_name="muted"),
                rx.upload(
                    rx.el.button("Choose a file", type="button", class_name="btn btn-accent"),
                    id="dataset_upload",
                    on_drop=State.handle_upload(rx.upload_files(upload_id="dataset_upload")),
                    multiple=False,
                ),
                class_name="panel panel-body",
            ),
        ),
    )
