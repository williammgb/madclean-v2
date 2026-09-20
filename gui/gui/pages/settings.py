"""The Settings view: everything about how a run behaves, in one place.

Reached from the gear in the top bar or "All settings" in the rail. The defaults come from
`configurations.json`, so what this view shows is what the command line would use too.
"""

from __future__ import annotations

import reflex as rx

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
                class_name="panel-body form-grid",
            ),
        ),
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
                        rx.el.option("Keep the cleaned column", value="accept"),
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
