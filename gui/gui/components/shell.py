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
    ("Settings", [("settings", "Settings", "sliders"), ("guide", "Guide", "guide")]),
]


def panel(*children, class_name: str = "panel", **props) -> rx.Component:
    return rx.el.div(*children, class_name=class_name, **props)


def panel_head(*children, **props) -> rx.Component:
    return rx.el.div(*children, class_name="panel-head", **props)


def modal(is_open, title, *children, footer: rx.Component) -> rx.Component:
    """A small dialog over the page, shown while `is_open` is true."""
    return rx.cond(
        is_open,
        rx.el.div(
            rx.el.div(
                rx.el.div(rx.el.h2(title), class_name="dlg-head"),
                rx.el.div(*children, class_name="modal-body"),
                rx.el.div(footer, class_name="dlg-foot"),
                class_name="modal",
                role="dialog",
                aria_modal="true",
            ),
            class_name="modal-scrim",
        ),
        rx.fragment(),
    )


def chip(*children, tone: str = "", large: bool = True) -> rx.Component:
    """One of the preview's pills. `tone` is green, red, orange, accent or empty."""
    classes = "chip" + (" chip-lg" if large else "") + (f" chip-{tone}" if tone else "")
    return rx.el.span(*children, class_name=classes)


def view_head(title: str, *extra) -> rx.Component:
    """A view's title and whatever sits beside it. No subtitle: a label names the view, and that is all."""
    return rx.el.div(rx.el.h1(title), *extra, class_name="view-head")


def switch(checked, on_change, *label, disabled=False) -> rx.Component:
    """The preview's toggle, with its label (if any) after the track."""
    return rx.el.label(
        rx.el.input(type="checkbox", checked=checked, on_change=on_change, disabled=disabled),
        rx.el.span(class_name="track"),
        *label,
        class_name="switch",
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
        rx.el.span(label, class_name="txt"),
        *trailing,
        type="button",
        title=label,
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
    collapsed = State.rail_collapsed == "1"
    return rx.el.aside(
        rx.el.nav(*groups, class_name="nav", aria_label="Workspace"),
        rx.el.div(
            run_setup(),
            rx.el.button(
                icon("collapse"),
                rx.el.span("Collapse", class_name="txt"),
                type="button",
                class_name="collapse-btn",
                aria_expanded=rx.cond(collapsed, "false", "true"),
                title=rx.cond(collapsed, "Expand sidebar", "Collapse sidebar"),
                on_click=State.toggle_rail,
            ),
            class_name="rail-foot",
        ),
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
    """The card at the bottom of the rail: one model per agent."""
    return rx.el.section(
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
        class_name="setup",
        aria_label="Models",
    )


def export_menu() -> rx.Component:
    """Export, and the three things a run can hand you to take away."""

    def item(label: str, icon_name: str, handler) -> rx.Component:
        return rx.el.button(
            icon(icon_name),
            label,
            type="button",
            on_click=[State.close_export_menu, handler],
        )

    return rx.el.div(
        rx.el.button(
            icon("download"),
            "Export",
            icon("down"),
            type="button",
            class_name="btn btn-soft",
            aria_expanded=rx.cond(State.export_menu_open, "true", "false"),
            on_click=State.toggle_export_menu,
        ),
        rx.cond(
            State.export_menu_open,
            rx.el.div(
                item("Cleaned data (.csv)", "table", State.download_cleaned_file),
                item("Cleaning code (.py)", "report", State.download_cleaning_code),
                item("Notebook (.ipynb)", "report", State.download_notebook),
                class_name="menu",
                role="menu",
            ),
            rx.fragment(),
        ),
        class_name="export",
    )


def top_bar() -> rx.Component:
    """Brand, the dataset, what the run is doing, and the run's own buttons."""
    return rx.el.header(
        rx.el.div(brand_mark(), "MADClean", class_name="brand"),
        rx.el.span(class_name="vsep"),
        rx.el.div(
            rx.el.span(icon("file"), State.file_name, class_name="file"),
            rx.el.span(State.dataset_meta, class_name="meta"),
            rx.cond(
                State.column_names.length() > 0,
                rx.upload(
                    rx.el.button(
                        icon("upload"), "Replace", type="button", class_name="btn btn-sm btn-ghost"
                    ),
                    id="dataset_replace",
                    on_drop=State.handle_upload(rx.upload_files(upload_id="dataset_replace")),
                    multiple=False,
                    no_keyboard=True,
                    # rx.upload draws a dashed drop zone with 5em of padding unless told not to.
                    border="none",
                    padding="0",
                ),
                rx.fragment(),
            ),
            class_name="dataset",
        ),
        rx.el.span(class_name="spacer"),
        rx.el.div(
            rx.el.span(
                rx.cond(State.last_run_status == "finished", icon("check"), rx.fragment()),
                State.run_chip_text,
                class_name=State.run_chip_class,
            ),
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
            rx.cond(State.has_cleaned, export_menu(), rx.fragment()),
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
