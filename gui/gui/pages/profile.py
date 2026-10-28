"""The Profile view: what profiling found, every column at once.

Each column is a card in a grid of two per row, so scrolling down gives the whole table's profile.
A card shows the column's counts, its most common values or a histogram, the shapes its values
take, and its findings: values that break a shape the rest follow, text in a number column,
numbers far from the rest, placeholder text and extra spaces. All of it is computed on this machine
before any model is called (see `profile_stats.py`).
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from ..components.shell import chip, panel, panel_head, view, view_head
from ..state import State


def stat(label: str, value, sub=None) -> rx.Component:
    return rx.el.div(
        rx.el.span(label, class_name="k"),
        rx.el.span(value, rx.el.small(sub) if sub is not None else rx.fragment(), class_name="v"),
        class_name="prof-stat",
    )


def histogram(row: rx.Var) -> rx.Component:
    return rx.el.div(
        rx.el.h3("Distribution"),
        rx.el.div(
            rx.foreach(
                row["histogram"].to(list[dict[str, str]]),
                lambda bar: rx.el.span(
                    style={"height": bar["height_pct"].to(str) + "%"},
                    title=bar["bin"],
                ),
            ),
            class_name="prof-hist",
            style={"--c": row["color"]},
        ),
        rx.el.div(
            rx.el.span(row["numeric_min"]),
            rx.el.span(row["numeric_max"]),
            class_name="hist-axis",
        ),
        class_name="prof-blk",
    )


def top_values(row: rx.Var) -> rx.Component:
    return rx.el.div(
        rx.el.h3("Most common values"),
        rx.el.div(
            rx.foreach(
                row["top_values"].to(list[dict[str, str]]),
                lambda value: rx.el.div(
                    rx.el.span(value["value"], class_name="lbl", title=value["value"]),
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
        class_name="prof-blk",
    )


def shapes(row: rx.Var) -> rx.Component:
    return rx.el.div(
        rx.el.h3("Shapes · ", row["shape_count"]),
        rx.el.div(
            rx.foreach(
                row["shapes"].to(list[dict[str, str]]),
                lambda shape: rx.el.div(
                    rx.el.span(shape["pattern"], class_name="lbl mono", title=shape["pattern"]),
                    rx.el.span(
                        rx.el.span(style={"width": shape["pct"].to(str) + "%"}),
                        class_name="track",
                    ),
                    rx.el.span(shape["pct"].to(str) + "%", class_name="n"),
                    class_name="bar-row",
                ),
            ),
            class_name="bars",
        ),
        class_name="prof-blk",
    )


def finding(item: rx.Var) -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.el.h3(item["title"]),
            rx.el.span(item["rule"], class_name="rule", title=item["rule"]),
            class_name="find-head",
        ),
        rx.el.div(
            rx.foreach(
                item["values"].to(list[dict[str, str]]),
                lambda value: rx.el.span(
                    rx.el.b(value["value"], title=value["value"]),
                    rx.el.i("×" + value["count"].to(str)),
                    class_name="val",
                ),
            ),
            class_name="vals",
        ),
        class_name="find",
    )


def column_card(row: rx.Var) -> rx.Component:
    return rx.el.article(
        rx.el.header(
            rx.el.span(class_name="type-dot", style={"--c": row["color"]}),
            rx.el.h2(row["column"], title=row["column"]),
            rx.el.span(row["semantic_type"], class_name="type-tag"),
            rx.cond(
                row["has_findings"] == "1",
                rx.el.span(
                    row["finding_count"],
                    rx.cond(row["finding_count"] == "1", " finding", " findings"),
                    class_name="chip chip-orange flag",
                ),
                rx.fragment(),
            ),
            class_name="prof-head",
        ),
        rx.el.div(
            stat("Missing", row["missing_count"], row["missing_pct"].to(str) + "%"),
            stat("Unique", row["unique_count"], row["unique_pct"].to(str) + "%"),
            stat("Repeated", row["repeated_count"], row["repeated_pct"].to(str) + "%"),
            stat("Length", row["length_range"], row["length_median"]),
            class_name="prof-stats",
        ),
        rx.cond(
            row["is_numeric"] == "1",
            rx.el.div(
                stat("Smallest", row["numeric_min"]),
                stat("Largest", row["numeric_max"]),
                stat("Mean", row["numeric_mean"]),
                stat("Median", row["numeric_median"]),
                class_name="prof-stats",
            ),
            rx.fragment(),
        ),
        rx.el.div(
            rx.cond(row["is_numeric"] == "1", histogram(row), top_values(row)),
            shapes(row),
            class_name="prof-body",
        ),
        rx.el.div(
            rx.cond(
                row["has_findings"] == "1",
                rx.foreach(row["findings"].to(list[dict[str, Any]]), finding),
                rx.el.span("No findings", class_name="clean"),
            ),
            class_name="prof-finds",
        ),
        class_name="panel prof-card",
    )


def filter_toggle() -> rx.Component:
    def option(value: str, label) -> rx.Component:
        return rx.el.button(
            label,
            type="button",
            aria_pressed=rx.cond(State.profile_filter == value, "true", "false"),
            on_click=State.set_profile_filter(value),
        )

    return rx.el.div(
        option("all", "All columns"),
        option("findings", "With findings"),
        class_name="seg",
        role="group",
        aria_label="Columns shown",
    )


def dependencies() -> rx.Component:
    return panel(
        panel_head(rx.el.h2("Functional dependencies")),
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
                rx.el.p("No dependencies", class_name="muted"),
                class_name="panel-body",
            ),
        ),
    )


def page() -> rx.Component:
    return view(
        "profile",
        view_head(
            "Profile",
            rx.el.div(
                chip(State.column_names.length().to(str) + " columns"),
                rx.cond(
                    State.profile_columns_with_findings > 0,
                    chip(
                        State.profile_columns_with_findings.to(str) + " with findings",
                        tone="orange",
                    ),
                    rx.fragment(),
                ),
                rx.cond(
                    State.fd_results.length() > 0,
                    chip(State.fd_results.length().to(str) + " dependencies", tone="accent"),
                    rx.fragment(),
                ),
                class_name="summary-row",
            ),
            rx.cond(
                State.profiling_dashboard_rows.length() > 0,
                rx.el.div(filter_toggle(), class_name="prof-tools"),
                rx.fragment(),
            ),
        ),
        rx.cond(
            State.profiling_dashboard_rows.length() > 0,
            rx.fragment(
                rx.el.div(
                    rx.foreach(
                        State.visible_profile_rows.to(list[dict[str, Any]]),
                        column_card,
                    ),
                    class_name="prof-grid",
                ),
                dependencies(),
            ),
            rx.el.div(rx.el.h2("No profile yet"), class_name="panel empty"),
        ),
    )
