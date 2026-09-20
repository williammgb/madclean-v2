"""The frame every view sits in: the top bar, the workflow rail, and the small pieces they share.

The markup is the approved preview's, class name for class name, so the stylesheet in
`gui/assets/madclean.css` styles it without a line of its own CSS here.
"""

from __future__ import annotations

import reflex as rx

from ..state import State
from .icons import brand_mark, icon

# The rail, in the preview's order: what you look at, what the run does, what comes out.
NAV_GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Data", [("table", "Table", "table"), ("profile", "Profile", "profile")]),
    (
        "Cleaning",
        [
            ("pipeline", "Pipeline", "flow"),
            ("review", "Review", "review"),
            ("logs", "Logs", "logs"),
        ],
    ),
    ("Results", [("report", "Report", "report"), ("evaluation", "Evaluation", "eval")]),
    ("Help", [("guide", "Guide", "guide")]),
]


def panel(*children, class_name: str = "panel", **props) -> rx.Component:
    return rx.el.div(*children, class_name=class_name, **props)


def panel_head(*children, **props) -> rx.Component:
    return rx.el.div(*children, class_name="panel-head", **props)


def panel_body(*children, **props) -> rx.Component:
    return rx.el.div(*children, class_name="panel-body", **props)


def chip(*children, tone: str = "", large: bool = True) -> rx.Component:
    """One of the preview's pills. `tone` is green, red, orange, accent or empty."""
    classes = "chip" + (" chip-lg" if large else "") + (f" chip-{tone}" if tone else "")
    return rx.el.span(*children, class_name=classes)


def view_head(title: str, subtitle: str, *extra) -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.el.h1(title),
            rx.el.p(subtitle, class_name="sub"),
        ),
        *extra,
        class_name="view-head",
    )


def empty(title: str, message, tone: str = "neutral") -> rx.Component:
    """The preview's empty state: a round icon, a heading and a sentence."""
    return rx.el.div(
        rx.el.div(icon("check" if tone != "neutral" else "minus"), class_name=f"empty-ic ic {tone}"),
        rx.el.h2(title),
        rx.el.p(message),
        class_name="empty",
    )


def nav_item(name: str, label: str, icon_name: str) -> rx.Component:
    """One entry in the rail. The count and the badge only appear where the preview has them."""
    trailing: list[rx.Component] = []
    if name == "pipeline":
        trailing.append(
            rx.cond(
                State.pipeline_task_count > 0,
                rx.el.span(State.pipeline_task_count, class_name="count"),
                rx.fragment(),
            )
        )
    if name == "review":
        trailing.append(
            rx.cond(
                State.review_waiting_count > 0,
                rx.el.span(State.review_waiting_count, class_name="alert"),
                rx.fragment(),
            )
        )
    return rx.el.button(
        icon(icon_name),
        label,
        *trailing,
        type="button",
        class_name=rx.cond(
            State.active_view == name, "nav-item is-active", "nav-item"
        ),
        on_click=State.show_view(name),
    )


def rail() -> rx.Component:
    groups = [
        rx.el.div(
            rx.el.div(label, class_name="nav-label"),
            *[nav_item(*item) for item in items],
            class_name="nav-group",
        )
        for label, items in NAV_GROUPS
    ]
    return rx.el.aside(
        rx.el.nav(*groups, class_name="nav", aria_label="Workspace"),
        run_setup(),
        class_name="rail",
    )


def model_row(label: str, value, on_change, options) -> rx.Component:
    return rx.el.div(
        rx.el.label(label),
        rx.el.select(
            rx.foreach(options, lambda name: rx.el.option(name, value=name)),
            value=value,
            on_change=on_change,
            class_name="select",
        ),
        class_name="setup-row",
    )


def run_setup() -> rx.Component:
    """The card at the bottom of the rail: one model per agent, and the human-in-the-loop switch."""
    return rx.el.section(
        rx.el.div(
            rx.el.h3("Run setup"),
            rx.el.button(
                "All settings",
                type="button",
                class_name="link small",
                on_click=State.show_view("settings"),
            ),
            class_name="setup-head",
        ),
        model_row(
            "Recommender",
            State.selected_llm_key_recommender,
            State.set_selected_llm_key_recommender,
            State.llm_options,
        ),
        model_row(
            "Coder",
            State.selected_llm_key_coding,
            State.set_selected_llm_key_coding,
            State.llm_options,
        ),
        model_row(
            "Validator",
            State.selected_llm_key_validation,
            State.set_selected_llm_key_validation,
            State.validation_llm_options,
        ),
        rx.el.div(
            rx.el.span("Human-in-the-loop", class_name="small"),
            rx.el.label(
                rx.el.input(
                    type="checkbox",
                    checked=State.human_in_the_loop,
                    on_change=State.set_human_in_the_loop,
                ),
                rx.el.span(class_name="track"),
                class_name="switch",
            ),
            class_name="switch-row",
        ),
        class_name="setup",
    )


def top_bar() -> rx.Component:
    """Brand, the dataset, what the run is doing, and the run's own buttons."""
    return rx.el.header(
        rx.el.div(brand_mark(), "MADClean", class_name="brand"),
        rx.el.span(class_name="vsep"),
        rx.el.div(
            rx.el.span(icon("file"), State.file_name, class_name="file"),
            rx.el.span(State.dataset_meta, class_name="meta"),
            class_name="dataset",
        ),
        rx.el.span(class_name="spacer"),
        rx.el.div(
            rx.el.span(State.run_chip_text, class_name=State.run_chip_class),
            rx.cond(
                State.is_cleaning,
                rx.el.span(
                    rx.el.span(style=State.progress_style),
                    class_name="progress",
                    role="progressbar",
                    aria_label="Run progress",
                ),
                rx.fragment(),
            ),
            class_name="status",
        ),
        rx.el.div(
            rx.el.button(
                icon("sliders"),
                type="button",
                class_name="btn btn-ghost btn-icon",
                aria_label="Settings",
                title="Settings",
                on_click=State.show_view("settings"),
            ),
            rx.cond(
                State.has_cleaned,
                rx.el.button(
                    icon("download"),
                    "Export",
                    type="button",
                    class_name="btn btn-soft",
                    on_click=State.show_view("report"),
                ),
                rx.fragment(),
            ),
            rx.cond(
                State.is_cleaning,
                rx.el.button(
                    icon("stop"),
                    "Stop run",
                    type="button",
                    class_name="btn btn-stop",
                    on_click=State.stop_pipeline,
                ),
                rx.el.button(
                    icon("play"),
                    rx.cond(State.has_cleaned, "Run again", "Run cleaning"),
                    type="button",
                    class_name="btn btn-run",
                    disabled=~State.column_names.length().bool(),
                    on_click=State.run_cleaning_process,
                ),
            ),
            class_name="actions",
        ),
        class_name="topbar",
    )


def view(name: str, *children) -> rx.Component:
    """One view of the main area, shown only when the rail points at it."""
    return rx.cond(
        State.active_view == name,
        rx.el.section(*children, class_name="view is-active", id=f"view-{name}"),
        rx.fragment(),
    )
