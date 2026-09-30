"""The Settings view: everything about how a run behaves, in one place.

Reached from the gear in the top bar or Settings in the rail. The defaults come from
`configurations.json`, so what this view shows is what the command line would use too, and
"Reset to defaults" reads that file again.
"""

from __future__ import annotations

import reflex as rx

from ..components.shell import panel_head, view, view_head
from ..state import State

# The sample-size matrix: a row per kind of column, and the two config keys under its two headers.
SAMPLE_ROWS = [
    ("Numeric", "NUMERIC", ("clean_sample_size", "dirty_sample_size")),
    ("Date and time", "DATETIME", ("clean_sample_size", "dirty_sample_size")),
    ("Dirty numeric", "DIRTY_NUMERIC", ("random_sample_size", "unique_sample_size")),
    ("Text", "STRING", ("random_sample_size", "unique_sample_size")),
]
FREE_TEXT_ROW = ("Free text", "NLT", ("short_sample_size", "long_sample_size"))


def card(title: str, *rows: rx.Component) -> rx.Component:
    return rx.el.section(
        panel_head(rx.el.h2(title)),
        rx.el.div(*rows, class_name="set-list"),
        class_name="panel",
        aria_label=title,
    )


def switch_row(title: str, name: str, checked, on_change) -> rx.Component:
    return rx.el.div(
        rx.el.label(title, html_for=name),
        rx.el.label(
            rx.el.input(type="checkbox", id=name, checked=checked, on_change=on_change),
            rx.el.span(class_name="track"),
            class_name="switch",
        ),
        class_name="set-row",
    )


def number_row(title: str, name: str, value, on_change, placeholder: str = "") -> rx.Component:
    return rx.el.div(
        rx.el.label(title, html_for=name),
        rx.el.input(
            id=name,
            value=value.to(str),
            on_change=on_change,
            class_name="input",
            input_mode="decimal" if placeholder else "numeric",
            placeholder=placeholder,
        ),
        class_name="set-row",
    )


def sample_input(label: str, category: str, key: str) -> rx.Component:
    return rx.el.td(
        rx.el.input(
            value=State.sample_sizes[category][key].to(str),
            on_change=lambda value: State.update_sample_size(category, key, value),
            class_name="input",
            input_mode="numeric",
            aria_label=label,
        )
    )


def sample_row(label: str, category: str, keys: tuple[str, str], headers: tuple[str, str]) -> rx.Component:
    return rx.el.tr(
        rx.el.td(label),
        *[sample_input(f"{label} {head.lower()}", category, key) for key, head in zip(keys, headers)],
    )


def samples() -> rx.Component:
    """How many example values of each kind of column the recommender is shown."""
    label, category, keys = FREE_TEXT_ROW
    return card(
        "Samples",
        rx.el.table(
            rx.el.thead(rx.el.tr(rx.el.th("Column type"), rx.el.th("Random"), rx.el.th("Dirty"))),
            rx.el.tbody(
                *[sample_row(name, cat, pair, ("Random", "Dirty")) for name, cat, pair in SAMPLE_ROWS]
            ),
            rx.el.tbody(
                rx.el.tr(rx.el.th(), rx.el.th("Short"), rx.el.th("Long"), class_name="sub"),
                sample_row(label, category, keys, ("Short", "Long")),
            ),
            class_name="matrix",
        ),
    )


def column_picks() -> rx.Component:
    """The columns human-in-the-loop asks about, when it is not asking about all of them."""
    everything = State.hitl_apply_to_all_columns
    return rx.cond(
        State.hitl_column_checkbox_rows.length() > 0,
        rx.el.div(
            rx.foreach(
                State.hitl_column_checkbox_rows,
                lambda row: rx.el.button(
                    row["column"],
                    type="button",
                    class_name="pick",
                    aria_pressed=rx.cond(~everything & (row["active"] == "1"), "true", "false"),
                    disabled=everything,
                    on_click=State.toggle_hitl_column_pick(row["column"]),
                ),
            ),
            class_name="col-picks",
        ),
        rx.fragment(),
    )


def page() -> rx.Component:
    return view(
        "settings",
        view_head(
            "Settings",
            rx.el.button(
                "Reset to defaults",
                type="button",
                class_name="btn btn-ghost btn-sm",
                on_click=State.reset_settings,
            ),
        ),
        rx.el.div(
            card(
                "Validation",
                switch_row("Validate columns", "set-validate", State.enable_validation, State.set_enable_validation),
                switch_row(
                    "Validate dependency tasks",
                    "set-validate-multi",
                    State.enable_validation_multi,
                    State.set_enable_validation_multi,
                ),
                rx.el.div(
                    rx.el.label("When undecided", html_for="set-undecided"),
                    rx.el.select(
                        rx.el.option("Keep the cleaned column", value="accept_cleaned"),
                        rx.el.option("Leave the column as it was", value="leave_uncleaned"),
                        rx.el.option("Ask me", value="ask_user"),
                        id="set-undecided",
                        value=State.validator_failure_strategy,
                        on_change=State.set_validator_failure_strategy,
                        class_name="select",
                    ),
                    class_name="set-row",
                ),
                number_row(
                    "Rows shown", "set-val-rows", State.sample_size_validator, State.set_sample_size_validator
                ),
                number_row(
                    "Random rows",
                    "set-val-random",
                    State.sample_size_validator_random,
                    State.set_sample_size_validator_random,
                ),
                number_row(
                    "Changed rows",
                    "set-val-changed",
                    State.sample_size_validator_changed,
                    State.set_sample_size_validator_changed,
                ),
            ),
            samples(),
            card(
                "Attempts",
                number_row("Per column", "set-att-col", State.max_cleaning_attempts, State.set_max_cleaning_attempts),
                number_row(
                    "Per dependency task",
                    "set-att-multi",
                    State.max_multi_col_attempts,
                    State.set_max_multi_col_attempts,
                ),
                number_row(
                    "Unreadable answer retries",
                    "set-att-parse",
                    State.max_parse_attempts,
                    State.set_max_parse_attempts,
                ),
                number_row(
                    "Failed code retries",
                    "set-att-code",
                    State.max_coding_attempts,
                    State.set_max_coding_attempts,
                ),
            ),
            card(
                "Dependencies",
                switch_row(
                    "Clean dependencies",
                    "set-deps",
                    State.enable_multi_col_cleaning,
                    State.set_enable_multi_col_cleaning,
                ),
            ),
            card(
                "Human review",
                switch_row("Human-in-the-loop", "set-hitl", State.human_in_the_loop, State.set_human_in_the_loop),
                switch_row(
                    "All columns",
                    "set-hitl-all",
                    State.hitl_apply_to_all_columns,
                    State.set_hitl_apply_to_all_columns,
                ),
                column_picks(),
            ),
            card(
                "Model",
                number_row(
                    "Temperature",
                    "set-temp",
                    State.llm_temperature_input,
                    State.set_llm_temperature_input,
                    placeholder="default",
                ),
                number_row(
                    "Top p", "set-top-p", State.llm_top_p_input, State.set_llm_top_p_input, placeholder="default"
                ),
                number_row("Columns at once", "set-sem", State.semaphore_limit, State.set_semaphore_limit),
                number_row(
                    "Labelled cells per column",
                    "set-labels",
                    State.max_labeled_cells_per_column,
                    State.set_max_labeled_cells_per_column,
                ),
            ),
            card("Output", switch_row("Verbose log", "set-verbose", State.verbose, State.set_verbose)),
            class_name="set-cols",
        ),
    )
