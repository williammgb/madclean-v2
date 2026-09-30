"""The Evaluation view: the run scored against a ground truth, if you have one.

With a file of the same columns and the same number of rows as the dataset, the run is scored cell
by cell: three scores, where the wrong cells went and what the run changed, and a row per column.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from ..components.icons import icon
from ..components.shell import panel, panel_head, view, view_head
from ..state import State


def score_tile(score: rx.Var) -> rx.Component:
    return rx.el.div(
        rx.el.span(score["k"], class_name="k"),
        rx.el.span(score["v"], rx.el.small(score["unit"]), class_name="v"),
        rx.el.span(
            rx.el.span(
                style={"width": score["width"].to(str), "background": score["colour"].to(str)}
            ),
            class_name="meter",
        ),
        class_name="panel score",
    )


def cell_bar(bar: rx.Var) -> rx.Component:
    """A count, the bar split by where those cells went, and the legend under it."""
    segments = bar["segments"].to(list[dict[str, str]])
    return rx.el.div(
        rx.el.div(
            rx.el.span(bar["label"], class_name="small muted"),
            rx.el.b(bar["total"]),
            class_name="top",
        ),
        rx.el.div(
            rx.foreach(
                segments,
                lambda seg: rx.cond(
                    seg["shown"] == "1",
                    rx.el.span(
                        style={
                            "flex": seg["flex"].to(str),
                            "minWidth": "4px",
                            "background": seg["colour"].to(str),
                        }
                    ),
                    rx.fragment(),
                ),
            ),
            class_name="split-bar",
        ),
        rx.el.div(
            rx.foreach(
                segments,
                lambda seg: rx.el.span(
                    rx.el.i(style={"background": seg["colour"].to(str)}),
                    seg["label"],
                    rx.el.b(seg["count"]),
                ),
            ),
            class_name="legend",
        ),
    )


def cells() -> rx.Component:
    return panel(
        panel_head(rx.el.h2("Cells")),
        rx.el.div(
            rx.foreach(State.evaluation_cell_bars.to(list[dict[str, Any]]), cell_bar),
            rx.el.div(
                rx.el.span("Correctly untouched", class_name="small muted"),
                rx.el.b(State.evaluation_untouched),
                class_name="untouched",
            ),
            class_name="cells",
        ),
    )


def column_row(row: rx.Var) -> rx.Component:
    return rx.el.tr(
        rx.el.td(
            rx.el.span(
                rx.el.span(class_name="type-dot", style={"--c": row["type_color"].to(str)}),
                row["column"],
                class_name="colname",
            )
        ),
        rx.el.td(
            rx.cond(
                row["has_f1"] == "1",
                rx.el.div(
                    rx.el.span(
                        rx.el.span(
                            style={
                                "width": row["f1_width"].to(str),
                                "background": row["f1_colour"].to(str),
                            }
                        ),
                        class_name="meter",
                    ),
                    rx.el.span(row["f1"], class_name=row["f1_class"].to(str)),
                    class_name="f1cell",
                ),
                rx.el.span("—", class_name="score-na"),
            )
        ),
        rx.el.td(row["precision"], class_name=row["precision_class"].to(str)),
        rx.el.td(row["recall"], class_name=row["recall_class"].to(str)),
        rx.el.td(row["repaired"], class_name=row["repaired_class"].to(str)),
        rx.el.td(row["fp"], class_name=row["fp_class"].to(str)),
        rx.el.td(row["fn"], class_name=row["fn_class"].to(str)),
    )


def per_column() -> rx.Component:
    return panel(
        panel_head(rx.el.h2("Per column")),
        rx.el.div(
            rx.el.table(
                rx.el.thead(
                    rx.el.tr(
                        rx.el.th("Column"),
                        rx.el.th("F1", style={"width": "34%"}),
                        rx.el.th("Precision", class_name="num"),
                        rx.el.th("Recall", class_name="num"),
                        rx.el.th("Repaired", class_name="num"),
                        rx.el.th("Changed but was fine", class_name="num"),
                        rx.el.th("Left wrong", class_name="num"),
                    )
                ),
                rx.el.tbody(
                    rx.foreach(State.evaluation_column_rows.to(list[dict[str, str]]), column_row)
                ),
                class_name="data",
            ),
            class_name="scroll-x",
        ),
    )


def ground_truth_bar() -> rx.Component:
    has_gt = State.gt_meta != ""
    return rx.el.div(
        rx.el.span(icon("file"), State.gt_file_name, class_name="file"),
        rx.cond(has_gt, rx.el.span(State.gt_meta, class_name="small muted"), rx.fragment()),
        rx.el.span(class_name="spacer"),
        rx.upload(
            rx.el.button(
                icon("upload"),
                rx.cond(has_gt, "Replace ground truth", "Upload ground truth"),
                type="button",
                class_name="btn btn-ghost btn-sm",
            ),
            id="gt_upload",
            on_drop=State.handle_gt_upload(rx.upload_files(upload_id="gt_upload")),
            multiple=False,
            no_keyboard=True,
            border="none",
            padding="0",
        ),
        rx.cond(
            has_gt,
            rx.el.button(
                icon("x"),
                type="button",
                class_name="btn btn-ghost btn-icon btn-sm",
                aria_label="Remove ground truth",
                title="Remove ground truth",
                on_click=State.clear_ground_truth,
            ),
            rx.fragment(),
        ),
        class_name="panel gt-bar",
    )


def page() -> rx.Component:
    return view(
        "evaluation",
        view_head("Evaluation"),
        ground_truth_bar(),
        rx.cond(
            State.evaluation_ready,
            rx.fragment(
                rx.el.div(
                    rx.foreach(State.evaluation_scores.to(list[dict[str, str]]), score_tile),
                    class_name="score-row",
                ),
                cells(),
                per_column(),
            ),
            rx.cond(
                State.evaluation_error != "",
                rx.el.div(
                    rx.el.div(icon("alert"), State.evaluation_error, class_name="callout"),
                    class_name="panel panel-body",
                ),
                rx.el.div(rx.el.h2("No scores yet"), class_name="panel empty"),
            ),
        ),
    )
