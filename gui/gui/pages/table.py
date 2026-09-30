"""The Table view: the data itself, with each column's semantic type in its header band.

Rows come from the state already prepared: every row carries the values to show, a flag per column
saying whether the cleaning changed that cell, and the value it had before. The switch above the
table picks what a changed cell shows: the value before, the value after, or both.

Every column has one width, set once in a `<colgroup>`, so a header and its values always line up.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from ..components.icons import icon
from ..components.shell import modal, switch, view, view_head
from ..state import State

TABLE_MODES = (("original", "Original"), ("cleaned", "Cleaned"), ("changes", "Changes"))


def column_header(col: rx.Var) -> rx.Component:
    """One header: the colour band on top, the name and its two buttons, the type underneath."""
    colour = State.display_column_header_colors[col]
    return rx.el.th(
        rx.el.div(
            rx.el.div(
                rx.el.span(col, class_name="th-name", title=col),
                rx.el.button(
                    icon("check"),
                    type="button",
                    class_name="th-btn",
                    aria_label="Mark " + col + " as already clean",
                    title="Mark as already clean",
                    aria_pressed=rx.cond(
                        State.user_marked_clean_columns[col].to(bool), "true", "false"
                    ),
                    on_click=State.toggle_user_marked_column_clean(col),
                ),
                rx.el.button(
                    icon("more"),
                    type="button",
                    class_name="th-btn",
                    aria_label="Instructions for " + col,
                    title="Instructions",
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
        scope="col",
        class_name=State.column_align[col].to(str),
        style={"--c": colour},
    )


def is_empty(value: rx.Var) -> rx.Var:
    """A missing value, however it reached the page: pandas' NaN arrives as the number NaN."""
    return (value == "") | (value == "nan") | (value == "NaN") | (value == "None")


def empty_or(value: rx.Var) -> rx.Component:
    return rx.cond(is_empty(value), rx.el.span("—", class_name="null"), rx.el.span(value))


def cell(row: rx.Var, col: rx.Var, index: rx.Var) -> rx.Component:
    """One cell. A changed cell shows what the switch asks for: before, after, or both.

    With "Label cells" on, a click opens the label dialog; a labelled cell carries its label.
    """
    value = row[col].to(str)
    changed = row["__modified_flags"].to(list[bool])[index]
    before = row["__original_values"].to(list[str])[index]
    kind = row["__label_kinds"].to(list[str])[index]
    shows_change = changed & (State.table_mode != "original")
    label_class = rx.cond(
        kind == "dirty", " lab lab-dirty", rx.cond(kind == "clean", " lab lab-clean", "")
    )
    return rx.el.td(
        rx.cond(
            shows_change,
            rx.fragment(
                rx.cond(
                    State.table_mode == "changes",
                    rx.el.span(rx.cond(is_empty(before), "—", before), class_name="old"),
                    rx.fragment(),
                ),
                rx.el.span(rx.cond(is_empty(value), "—", value), class_name="new"),
            ),
            empty_or(rx.cond(changed, before, value)),
        ),
        rx.cond(kind != "", rx.el.span(kind, class_name="lab-tag"), rx.fragment()),
        class_name=State.column_align[col].to(str)
        + rx.cond(shows_change, " chg", "")
        + label_class,
        title=rx.cond(shows_change, before + " → " + value, rx.cond(changed, before, value)),
        on_click=State.open_cell_label_dialog(row["row_id"].to(str), col, value),
    )


def grid() -> rx.Component:
    return rx.el.div(
        rx.el.table(
            rx.el.colgroup(
                rx.el.col(style={"width": "52px"}),
                rx.foreach(
                    State.column_names,
                    lambda col: rx.el.col(style={"width": State.column_widths[col].to(str)}),
                ),
            ),
            rx.el.thead(
                rx.el.tr(
                    rx.el.th("#", scope="col", class_name="rownum"),
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


def mode_switch() -> rx.Component:
    """Original / Cleaned / Changes, once a run has cleaned something."""
    return rx.cond(
        State.has_cleaned,
        rx.el.div(
            *[
                rx.el.button(
                    label,
                    type="button",
                    aria_pressed=rx.cond(State.table_mode == mode, "true", "false"),
                    on_click=State.set_table_mode(mode),
                )
                for mode, label in TABLE_MODES
            ],
            class_name="seg",
            role="group",
            aria_label="Table version",
        ),
        rx.fragment(),
    )


def toolbar() -> rx.Component:
    return rx.el.div(
        mode_switch(),
        switch(State.show_fds, State.set_show_fds, "Dependencies"),
        switch(State.label_cells_mode, State.set_label_cells_mode, "Label cells"),
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
            rx.el.span("Dependencies", class_name="small muted"),
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


def hint_dialog() -> rx.Component:
    """Instructions for one column, added to the recommender's prompt for that column only."""
    return modal(
        State.recommender_hint_dialog_open,
        "Instructions for " + State.recommender_hint_editing_column,
        rx.el.textarea(
            value=State.recommender_hint_editor_text,
            on_change=State.set_recommender_hint_editor_text,
            class_name="input",
            rows="6",
            aria_label="Instructions",
        ),
        footer=rx.fragment(
            rx.el.button(
                "Cancel",
                type="button",
                class_name="btn btn-ghost",
                on_click=State.cancel_recommender_hint_editor,
            ),
            rx.el.button(
                "Save",
                type="button",
                class_name="btn btn-accent",
                on_click=State.save_recommender_hint_and_close,
            ),
        ),
    )


def label_dialog() -> rx.Component:
    """Label one cell clean or dirty; labels reach the recommender as worked examples."""
    current = State.labeling_current_value_preview
    return modal(
        State.labeling_dialog_open,
        "Label a cell in " + State.labeling_target_col,
        rx.el.div(rx.cond(current == "", "—", current), class_name="value"),
        rx.el.div(
            rx.el.button(
                "Clean",
                type="button",
                aria_pressed=rx.cond(State.labeling_kind == "clean", "true", "false"),
                on_click=State.set_labeling_kind("clean"),
            ),
            rx.el.button(
                "Dirty",
                type="button",
                aria_pressed=rx.cond(State.labeling_kind == "dirty", "true", "false"),
                on_click=State.set_labeling_kind("dirty"),
            ),
            class_name="seg",
            role="group",
            aria_label="Label",
        ),
        rx.cond(
            State.labeling_kind == "dirty",
            rx.el.div(
                rx.el.span("Correct value", class_name="lbl"),
                rx.el.textarea(
                    value=State.labeling_expected_value,
                    on_change=State.set_labeling_expected_value,
                    class_name="input",
                    rows="3",
                ),
                class_name="field",
            ),
            rx.fragment(),
        ),
        footer=rx.fragment(
            rx.el.button(
                "Remove label",
                type="button",
                class_name="btn btn-orange",
                on_click=State.clear_current_cell_label,
            ),
            rx.el.span(class_name="spacer"),
            rx.el.button(
                "Cancel",
                type="button",
                class_name="btn btn-ghost",
                on_click=State.cancel_cell_label_dialog,
            ),
            rx.el.button(
                "Save",
                type="button",
                class_name="btn btn-accent",
                on_click=State.save_cell_label,
            ),
        ),
    )


def page() -> rx.Component:
    # The wrapper's classes switch on the dependency badges in the headers and the labelling cursor.
    mode_class = rx.cond(State.show_fds, "show-fd", "") + rx.cond(
        State.label_cells_mode, " labeling", ""
    )
    return view(
        "table",
        view_head("Table"),
        rx.cond(
            State.column_names.length() > 0,
            rx.el.div(
                toolbar(),
                dependency_strip(),
                grid(),
                rx.cond(
                    State.has_cleaned,
                    rx.el.div(State.changed_cells_label, class_name="table-foot"),
                    rx.fragment(),
                ),
                hint_dialog(),
                label_dialog(),
                class_name=mode_class,
                style={"display": "flex", "flexDirection": "column", "gap": "12px"},
            ),
            rx.el.div(
                rx.el.h2("No dataset yet"),
                rx.upload(
                    rx.el.button(
                        icon("upload"), "Choose a file", type="button", class_name="btn btn-accent"
                    ),
                    id="dataset_upload",
                    on_drop=State.handle_upload(rx.upload_files(upload_id="dataset_upload")),
                    multiple=False,
                ),
                class_name="panel empty",
            ),
        ),
    )
