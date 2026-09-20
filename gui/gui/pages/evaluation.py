"""The Evaluation view: the run scored against a ground truth, if you have one.

Upload a file with the same columns and the same number of rows as the dataset, and the run is
scored cell by cell: which wrong cells it found, and which of them it repaired correctly.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from ..components.icons import icon
from ..components.shell import panel, panel_head, view, view_head
from ..state import State


def score_block(title: str, rows: rx.Var) -> rx.Component:
    return panel(
        panel_head(rx.el.h2(title)),
        rx.foreach(
            rows.to(list[dict[str, str]]),
            lambda row: rx.fragment(
                rx.el.div(
                    rx.el.div(rx.el.span("Precision", class_name="k"), rx.el.span(row["precision"], class_name="v")),
                    rx.el.div(rx.el.span("Recall", class_name="k"), rx.el.span(row["recall"], class_name="v")),
                    rx.el.div(rx.el.span("F1", class_name="k"), rx.el.span(row["f1"], class_name="v")),
                    class_name="prf",
                ),
                rx.el.div(
                    rx.el.div(rx.el.span("Found and repaired", class_name="k"), rx.el.span(row["tp"], class_name="v")),
                    rx.el.div(rx.el.span("Changed but was fine", class_name="k"), rx.el.span(row["fp"], class_name="v")),
                    rx.el.div(rx.el.span("Left wrong", class_name="k"), rx.el.span(row["fn"], class_name="v")),
                    rx.el.div(rx.el.span("Correctly untouched", class_name="k"), rx.el.span(row["tn"], class_name="v")),
                    class_name="counts",
                ),
            ),
        ),
    )


def per_column() -> rx.Component:
    return panel(
        panel_head(rx.el.h2("Per column")),
        rx.el.div(
            rx.el.table(
                rx.el.thead(
                    rx.el.tr(
                        rx.el.th("Column"),
                        rx.el.th("Precision", class_name="num"),
                        rx.el.th("Recall", class_name="num"),
                        rx.el.th("F1", class_name="num"),
                        rx.el.th("Found", class_name="num bl"),
                        rx.el.th("Wrongly changed", class_name="num"),
                        rx.el.th("Left wrong", class_name="num"),
                    )
                ),
                rx.el.tbody(
                    rx.foreach(
                        State.evaluation_column_rows.to(list[dict[str, str]]),
                        lambda row: rx.el.tr(
                            rx.el.td(rx.el.b(row["column"])),
                            rx.el.td(row["precision"], class_name="num"),
                            rx.el.td(row["recall"], class_name="num"),
                            rx.el.td(row["f1"], class_name="num"),
                            rx.el.td(row["tp"], class_name="num bl"),
                            rx.el.td(row["fp"], class_name="num"),
                            rx.el.td(row["fn"], class_name="num"),
                        ),
                    )
                ),
                class_name="data",
            ),
            class_name="scroll-x",
        ),
    )


def ground_truth_bar() -> rx.Component:
    return panel(
        rx.el.div(
            rx.el.span(icon("file"), State.gt_file_name, class_name="file"),
            rx.upload(
                rx.el.button(
                    icon("upload"),
                    "Upload ground truth",
                    type="button",
                    class_name="btn btn-ghost btn-sm",
                ),
                id="gt_upload",
                on_drop=State.handle_gt_upload(rx.upload_files(upload_id="gt_upload")),
                multiple=False,
            ),
            rx.el.span(class_name="spacer"),
            rx.el.span(State.evaluation_status, class_name="small muted"),
            class_name="gt-bar",
        )
    )


def page() -> rx.Component:
    return view(
        "evaluation",
        view_head(
            "Evaluation",
            "A ground-truth file with the same columns and row count turns the run into scores: "
            "what it found, and what it repaired correctly.",
        ),
        ground_truth_bar(),
        rx.cond(
            State.evaluation_ready,
            rx.fragment(
                rx.el.div(
                    score_block("This run", State.evaluation_overall_rows),
                    class_name="metric-pair",
                ),
                per_column(),
            ),
            rx.cond(
                State.evaluation_error != "",
                rx.el.div(
                    rx.el.div(icon("alert"), State.evaluation_error, class_name="callout"),
                    class_name="panel panel-body",
                ),
                rx.fragment(),
            ),
        ),
    )
