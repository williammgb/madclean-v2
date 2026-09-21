"""The Settings view: everything about how a run behaves, in one place.

Reached from the gear in the top bar or "All settings" in the rail. The defaults come from
`configurations.json`, so what this view shows is what the command line would use too.
"""

from __future__ import annotations

import reflex as rx

from madclean.config.loader import load_default_cleaning_config

from ..components.shell import panel, panel_head, view, view_head
from ..state import State


def switch_row(title: str, description: str, checked, on_change) -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.el.div(title, class_name="t"),
            rx.el.div(description, class_name="d"),
        ),
        rx.el.label(
            rx.el.input(type="checkbox", checked=checked, on_change=on_change),
            rx.el.span(class_name="track"),
            class_name="switch",
        ),
        class_name="set-row",
    )


def number_field(label: str, value, on_change, note: str = "") -> rx.Component:
    return rx.el.div(
        rx.el.label(label),
        rx.el.input(
            value=value.to(str),
            on_change=on_change,
            class_name="input",
            input_mode="numeric",
        ),
        rx.cond(note != "", rx.el.span(note, class_name="def"), rx.fragment()),
        class_name="num-field",
    )


def text_field(label: str, value, on_change, note: str = "") -> rx.Component:
    """A free-text field; empty means "use the default"."""
    return rx.el.div(
        rx.el.label(label),
        rx.el.input(
            value=value,
            on_change=on_change,
            class_name="input",
            placeholder="default",
            input_mode="decimal",
        ),
        rx.cond(note != "", rx.el.span(note, class_name="def"), rx.fragment()),
        class_name="num-field",
    )


def hitl_column_picker() -> rx.Component:
    """The columns human-in-the-loop asks about, when it is not asking about all of them."""
    return rx.el.div(
        rx.el.span("Columns to ask about", class_name="lbl"),
        rx.cond(
            State.hitl_column_checkbox_rows.length() > 0,
            rx.el.div(
                rx.foreach(
                    State.hitl_column_checkbox_rows,
                    lambda row: rx.el.button(
                        row["column"],
                        type="button",
                        class_name=rx.cond(row["active"] == "1", "fd-chip is-on", "fd-chip"),
                        aria_pressed=rx.cond(row["active"] == "1", "true", "false"),
                        on_click=State.toggle_hitl_column_pick(row["column"]),
                    ),
                ),
                class_name="pick-list",
            ),
            rx.el.span("Upload a dataset to pick its columns.", class_name="small muted"),
        ),
        class_name="field",
    )


def sample_size_field(category: str, key: str) -> rx.Component:
    label = f"{category.replace('_', ' ').title()}: {key.replace('_sample_size', '')} values"
    return number_field(
        label,
        State.sample_sizes[category][key],
        lambda value: State.update_sample_size(category, key, value),
    )


def sample_sizes() -> rx.Component:
    """How many example values of each kind of column the recommender is shown."""
    fields = [
        sample_size_field(category, key)
        for category, sizes in load_default_cleaning_config().sample_sizes.items()
        for key in sizes
    ]
    return panel(
        panel_head(
            rx.el.div(
                rx.el.h2("What the recommender sees"),
                rx.el.p(
                    "How many example values are sent to the model for each kind of column.",
                    class_name="small muted",
                ),
            )
        ),
        rx.el.div(*fields, class_name="panel-body form-grid"),
    )


def page() -> rx.Component:
    return view(
        "settings",
        view_head(
            "Settings",
            "How a run behaves. These are the same settings the command line reads from "
            "configurations.json, so a run started here and a run started there do the same thing.",
        ),
        panel(
            panel_head(rx.el.h2("What runs")),
            rx.el.div(
                switch_row(
                    "Validate each cleaned column",
                    "The validator agent reads the cleaned column against the original and can "
                    "send it back for another attempt.",
                    State.enable_validation,
                    State.set_enable_validation,
                ),
                switch_row(
                    "Clean dependencies between columns",
                    "After the columns, rules such as \"this brewery id always has this state\" "
                    "become their own cleaning tasks.",
                    State.enable_multi_col_cleaning,
                    State.set_enable_multi_col_cleaning,
                ),
                switch_row(
                    "Validate dependency tasks too",
                    "Off by default, as in the thesis: dependency tasks are applied without a "
                    "validation pass.",
                    State.enable_validation_multi,
                    State.set_enable_validation_multi,
                ),
                switch_row(
                    "Human-in-the-loop",
                    "The run stops and asks you before it applies code, and when the validator "
                    "disagrees with the cleaning.",
                    State.human_in_the_loop,
                    State.set_human_in_the_loop,
                ),
                switch_row(
                    "Ask about every column",
                    "With human-in-the-loop on: ask about all columns, rather than only the ones "
                    "you picked.",
                    State.hitl_apply_to_all_columns,
                    State.set_hitl_apply_to_all_columns,
                ),
                rx.cond(
                    State.human_in_the_loop & ~State.hitl_apply_to_all_columns,
                    hitl_column_picker(),
                    rx.fragment(),
                ),
                switch_row(
                    "Verbose output",
                    "Print each task's progress to the log as it happens.",
                    State.verbose,
                    State.set_verbose,
                ),
                class_name="panel-body dlg-sec",
            ),
        ),
        panel(
            panel_head(
                rx.el.div(
                    rx.el.h2("How hard it tries"),
                    rx.el.p(
                        "An attempt is one trip around the agents: recommend, write code, "
                        "validate.",
                        class_name="small muted",
                    ),
                )
            ),
            rx.el.div(
                number_field(
                    "Attempts per column",
                    State.max_cleaning_attempts,
                    State.set_max_cleaning_attempts,
                ),
                number_field(
                    "Attempts per dependency task",
                    State.max_multi_col_attempts,
                    State.set_max_multi_col_attempts,
                ),
                number_field(
                    "Retries when the model's answer will not parse",
                    State.max_parse_attempts,
                    State.set_max_parse_attempts,
                ),
                number_field(
                    "Retries when the generated code fails to run",
                    State.max_coding_attempts,
                    State.set_max_coding_attempts,
                ),
                number_field(
                    "Columns cleaned at once",
                    State.semaphore_limit,
                    State.set_semaphore_limit,
                    "15 in the thesis runs",
                ),
                number_field(
                    "Labelled cells sent per column",
                    State.max_labeled_cells_per_column,
                    State.set_max_labeled_cells_per_column,
                    "From the Table's label mode",
                ),
                text_field(
                    "Model temperature",
                    State.llm_temperature_input,
                    State.set_llm_temperature_input,
                    "Empty uses the provider's default",
                ),
                text_field(
                    "Model top_p",
                    State.llm_top_p_input,
                    State.set_llm_top_p_input,
                    "Empty uses the provider's default",
                ),
                class_name="panel-body form-grid",
            ),
        ),
        sample_sizes(),
        panel(
            panel_head(
                rx.el.div(
                    rx.el.h2("What the validator sees"),
                    rx.el.p(
                        "How many rows are shown to the validator, and what happens when it "
                        "cannot decide.",
                        class_name="small muted",
                    ),
                )
            ),
            rx.el.div(
                number_field(
                    "Rows shown",
                    State.sample_size_validator,
                    State.set_sample_size_validator,
                ),
                number_field(
                    "Of those, chosen at random",
                    State.sample_size_validator_random,
                    State.set_sample_size_validator_random,
                ),
                number_field(
                    "Of those, cells the cleaning changed",
                    State.sample_size_validator_changed,
                    State.set_sample_size_validator_changed,
                ),
                rx.el.div(
                    rx.el.label("When the validator cannot decide"),
                    rx.el.select(
                        rx.el.option("Keep the cleaned column", value="accept_cleaned"),
                        rx.el.option("Leave the column as it was", value="leave_uncleaned"),
                        rx.el.option("Ask me", value="ask_user"),
                        value=State.validator_failure_strategy,
                        on_change=State.set_validator_failure_strategy,
                        class_name="select",
                    ),
                    class_name="num-field",
                ),
                class_name="panel-body form-grid",
            ),
        ),
    )
