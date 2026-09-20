"""The Profile view: what profiling found, column by column.

All of it is computed on this machine before any model is called — the semantic type, the missing
and unique counts, the shape of the numbers, the most common values and patterns, and the
dependencies between columns.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from ..components.icons import icon
from ..components.shell import chip, panel, panel_head, view, view_head
from ..state import State


def column_button(row: rx.Var) -> rx.Component:
    """One column in the list on the left, with how much of it is missing."""
    name = row["column"]
    return rx.el.button(
        rx.el.span(class_name="type-dot", style={"--c": row["color"]}),
        rx.el.span(name, class_name="nm"),
        rx.el.span(row["semantic_type"], class_name="ty"),
        rx.el.span(
            rx.el.span(row["missing_pct"].to(str) + "%"),
            rx.el.span(
                rx.el.span(style={"width": row["missing_pct"].to(str) + "%"}),
                class_name="minibar",
            ),
            class_name="miss",
        ),
        type="button",
        class_name="col-btn",
        aria_pressed=rx.cond(State.selected_profile_column == name, "true", "false"),
        on_click=State.set_selected_profile_column(name),
    )


def stat(label: str, value) -> rx.Component:
    return rx.el.div(
        rx.el.span(label, class_name="k"),
        rx.el.span(value, class_name="v"),
        class_name="stat",
    )


def histogram(row: rx.Var) -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.foreach(
                row["histogram"].to(list[dict[str, Any]]),
                lambda bar: rx.el.span(
                    style={"height": bar["height_pct"].to(str) + "%"},
                    title=bar["bin"],
                ),
            ),
            class_name="hist",
            style={"--c": row["color"]},
        ),
        rx.el.div(
            rx.el.span(row["hist_x_min"]),
            rx.el.span(row["hist_x_max"]),
            class_name="hist-axis",
        ),
        class_name="block",
    )


def top_values(row: rx.Var) -> rx.Component:
    return rx.el.div(
        rx.el.h3("Most common values"),
        rx.el.div(
            rx.foreach(
                row["top_values"].to(list[dict[str, str]]),
                lambda value: rx.el.div(
                    rx.el.span(value["value"], class_name="lbl"),
                    rx.el.span(
                        rx.el.span(style={"width": value["pct"].to(str) + "%"}),
                        class_name="track",
                    ),
                    rx.el.span(value["count"], class_name="n"),
                    class_name="bar-row",
                    style={"--c": row["color"]},
                ),
            ),
            class_name="bars",
        ),
        class_name="block",
    )


def patterns(row: rx.Var) -> rx.Component:
    return rx.cond(
        row["has_top_regex"] == "1",
        rx.el.div(
            rx.el.h3("Shapes the values take"),
            rx.el.div(
                rx.foreach(
                    row["top_regex"].to(list[dict[str, str]]),
                    lambda pattern: rx.el.div(
                        rx.el.span(pattern["pattern"], class_name="pattern"),
                        rx.el.span(
                            rx.el.span(style={"width": pattern["pct"].to(str) + "%"}),
                            class_name="track",
                        ),
                        rx.el.span(pattern["pct"].to(str) + "%", class_name="n"),
                        class_name="bar-row",
                    ),
                ),
                class_name="bars",
            ),
            class_name="block",
        ),
        rx.fragment(),
    )


def column_detail(row: rx.Var) -> rx.Component:
    return rx.fragment(
        panel_head(
            rx.el.div(
                rx.el.h2(row["column"]),
                rx.el.span(
                    rx.el.span(class_name="type-dot", style={"--c": row["color"]}),
                    row["semantic_type"],
                    class_name="type-tag",
                ),
            ),
        ),
        rx.el.div(
            rx.el.div(
                stat(
                    "Missing",
                    row["missing_count"].to(str) + " (" + row["missing_pct"].to(str) + "%)",
                ),
                stat(
                    "Unique",
                    row["unique_count"].to(str) + " (" + row["unique_pct"].to(str) + "%)",
                ),
                rx.cond(
                    row["is_numeric"] == "1",
                    rx.fragment(
                        stat("Smallest", row["numeric_min"]),
                        stat("Largest", row["numeric_max"]),
                        stat("Mean", row["numeric_mean"]),
                        stat("Median", row["numeric_median"]),
                    ),
                    rx.fragment(),
                ),
                class_name="stat-grid",
            ),
            rx.el.div(
                rx.cond(row["is_numeric"] == "1", histogram(row), rx.fragment()),
                top_values(row),
                patterns(row),
                class_name="detail-grid",
            ),
            class_name="panel-body",
            style={"display": "flex", "flexDirection": "column", "gap": "18px"},
        ),
    )


def dependencies() -> rx.Component:
    return panel(
        panel_head(
            rx.el.div(
                rx.el.h2("Functional dependencies"),
                rx.el.p(
                    "Pairs where one column almost always decides the other (score of at least "
                    "0.925). These become multi-column cleaning tasks.",
                    class_name="small muted",
                ),
            )
        ),
        rx.cond(
            State.fd_results.length() > 0,
            rx.el.div(
                rx.el.table(
                    rx.el.thead(
                        rx.el.tr(
                            rx.el.th("Rule"),
                            rx.el.th("Score", class_name="num"),
                            rx.el.th("Conflicting groups", class_name="num"),
                            rx.el.th("Blanks that can be filled", class_name="num"),
                        )
                    ),
                    rx.el.tbody(
                        rx.foreach(
                            State.fd_results.to(list[dict[str, Any]]),
                            lambda fd: rx.el.tr(
                                rx.el.td(
                                    rx.el.b(fd["lhs"]), " → ", rx.el.b(fd["rhs"])
                                ),
                                rx.el.td(fd["score"].to(str), class_name="num"),
                                rx.el.td(fd["violations_count"].to(str), class_name="num"),
                                rx.el.td(fd["imputables_count"].to(str), class_name="num"),
                            ),
                        )
                    ),
                    class_name="data",
                ),
                class_name="scroll-x",
            ),
            rx.el.div(
                rx.el.p("No dependencies were found in this dataset.", class_name="muted"),
                class_name="panel-body",
            ),
        ),
    )


def page() -> rx.Component:
    return view(
        "profile",
        view_head(
            "Profile",
            "Computed on this machine from every row. Nothing is sent to a model at this step.",
            rx.el.div(
                chip(State.column_names.length().to(str) + " columns"),
                rx.cond(
                    State.fd_results.length() > 0,
                    chip(State.fd_results.length().to(str) + " dependencies", tone="accent"),
                    rx.fragment(),
                ),
                class_name="summary-row",
            ),
        ),
        rx.cond(
            State.profiling_dashboard_rows.length() > 0,
            rx.fragment(
                rx.el.div(
                    rx.el.div(
                        rx.foreach(
                            State.profiling_dashboard_rows.to(list[dict[str, Any]]),
                            column_button,
                        ),
                        class_name="panel col-list",
                        role="group",
                        aria_label="Columns",
                    ),
                    panel(
                        rx.foreach(
                            State.selected_profile_rows.to(list[dict[str, Any]]),
                            column_detail,
                        )
                    ),
                    class_name="split",
                ),
                dependencies(),
            ),
            rx.el.div(
                rx.el.div(
                    icon("info"),
                    "Upload a dataset and profiling runs by itself.",
                    class_name="callout info",
                ),
                class_name="panel panel-body",
            ),
        ),
    )
