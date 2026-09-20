"""The Review view: the decisions the run is waiting on, and the answers you give it.

With human-in-the-loop on, or the validator set to USER, the run stops on a decision and waits.
The rest of the run keeps going. Each question shows what the agents did and what they are asking
for, and the buttons here are the answer the waiting task receives.

Every value read out of a row is typed on the way out (`.to(str)` and friends): the rows are
dictionaries of mixed content, and the page cannot render a value whose type it does not know.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from ..components.shell import chip, panel, panel_head, view, view_head
from ..state import State


def text(row: rx.Var, key: str) -> rx.Var:
    return row[key].to(str)


def lines(row: rx.Var, key: str) -> rx.Var:
    return row[key].to(list[str])


def samples(row: rx.Var, key: str) -> rx.Var:
    return row[key].to(list[dict[str, str]])


def sample_table(rows: rx.Var) -> rx.Component:
    """The before-and-after rows the question is about."""
    return rx.el.div(
        rx.el.table(
            rx.el.thead(rx.el.tr(rx.el.th("Original"), rx.el.th("Cleaned"))),
            rx.el.tbody(
                rx.foreach(
                    rows,
                    lambda sample: rx.el.tr(
                        rx.el.td(sample["original"].to(str), class_name="o"),
                        rx.el.td(sample["cleaned"].to(str), class_name="c"),
                    ),
                )
            ),
            class_name="data compare",
        ),
        class_name="scroll-x",
    )


def queue_entry(row: rx.Var) -> rx.Component:
    """One question in the list on the left."""
    column = text(row, "column")
    return rx.el.button(
        rx.el.div(
            rx.el.b(column),
            chip(text(row, "hitl_kind"), tone="orange", large=False),
            class_name="row",
        ),
        rx.el.span(text(row, "active_title"), class_name="small muted"),
        type="button",
        class_name="queue-item",
        aria_pressed=rx.cond(State.pending_validation_column == column, "true", "false"),
        on_click=State.select_pipeline_flow_step(column, text(row, "expanded_step_id")),
    )


def code_review(row: rx.Var) -> rx.Component:
    """The coder wrote this; you can edit it before it runs."""
    column = text(row, "column")
    return rx.el.div(
        rx.el.div(
            rx.el.span("The code the coder wrote", class_name="lbl"),
            rx.el.textarea(
                value=text(row, "hitl_code"),
                on_change=lambda value: State.set_hitl_code_column(column, value),
                class_name="input mono",
                rows="14",
            ),
            class_name="field",
        ),
        rx.el.div(
            rx.el.button(
                "Run this code",
                type="button",
                class_name="btn btn-accent",
                on_click=State.submit_hitl_code_review(text(row, "hitl_request_id"), column),
            ),
            class_name="choices",
        ),
        class_name="decision",
    )


def validation_review(row: rx.Var) -> rx.Component:
    """The validator reached a verdict; you agree with it or overrule it."""
    column = text(row, "column")
    request_id = text(row, "hitl_request_id")
    return rx.el.div(
        rx.el.ul(
            rx.foreach(lines(row, "hitl_validator_summary_lines"), lambda line: rx.el.li(line))
        ),
        sample_table(samples(row, "hitl_sample_rows")),
        rx.el.div(
            rx.el.span("Your instructions, if you want another attempt", class_name="lbl"),
            rx.el.textarea(
                value=text(row, "hitl_feedback_message"),
                on_change=lambda value: State.set_hitl_validation_feedback_message_column(
                    column, value
                ),
                class_name="input",
                placeholder="What should be done differently?",
            ),
            class_name="field",
        ),
        rx.el.div(
            rx.el.button(
                "Agree with the validator",
                type="button",
                class_name="btn btn-green",
                on_click=State.submit_hitl_validation_validator_ok_agree(request_id, column),
            ),
            rx.el.button(
                "Accept the feedback and try again",
                type="button",
                class_name="btn btn-orange",
                on_click=State.submit_hitl_validation_feedback_accept(request_id, column),
            ),
            rx.el.button(
                "The cleaning is fine, keep it",
                type="button",
                class_name="btn btn-ghost",
                on_click=State.submit_hitl_validation_feedback_reject_cleaning_valid(
                    request_id, column
                ),
            ),
            class_name="choices",
        ),
        class_name="decision",
    )


def already_clean_review(row: rx.Var) -> rx.Component:
    """The recommender says the column needs nothing; you confirm or send it back."""
    column = text(row, "column")
    request_id = text(row, "hitl_request_id")
    return rx.el.div(
        rx.el.ul(
            rx.foreach(lines(row, "hitl_validator_summary_lines"), lambda line: rx.el.li(line))
        ),
        rx.el.div(
            rx.el.button(
                "Agreed, leave it alone",
                type="button",
                class_name="btn btn-green",
                on_click=State.submit_hitl_already_clean_confirm(request_id, column),
            ),
            rx.el.button(
                "No, it does need cleaning",
                type="button",
                class_name="btn btn-orange",
                on_click=State.submit_hitl_already_clean_reject(request_id, column),
            ),
            class_name="choices",
        ),
        class_name="decision",
    )


def user_validation() -> rx.Component:
    """The validator seat, when the validator is set to USER."""
    return panel(
        panel_head(
            rx.el.div(
                rx.el.h2("Validate " + State.pending_validation_column),
                rx.el.p(
                    "Attempt "
                    + State.pending_validation_attempt.to(str)
                    + " · "
                    + State.pending_validation_modified_count.to(str)
                    + " cells changed",
                    class_name="small muted",
                ),
            )
        ),
        rx.el.div(
            sample_table(State.pending_validation_sample_rows),
            rx.el.div(
                rx.el.span("What should be done differently?", class_name="lbl"),
                rx.el.textarea(
                    value=State.user_validation_feedback_message,
                    on_change=State.set_user_validation_feedback_message,
                    class_name="input",
                ),
                class_name="field",
            ),
            rx.el.div(
                rx.el.button(
                    "Accept the cleaning",
                    type="button",
                    class_name="btn btn-green",
                    on_click=State.accept_user_validation(State.pending_validation_column),
                ),
                rx.el.button(
                    "Send it back",
                    type="button",
                    class_name="btn btn-orange",
                    on_click=State.submit_user_validation(State.pending_validation_column),
                ),
                class_name="choices",
            ),
            class_name="panel-body decision",
        ),
    )


def question(row: rx.Var) -> rx.Component:
    kind = text(row, "hitl_kind")
    return panel(
        panel_head(
            rx.el.div(
                rx.el.h2(text(row, "column")),
                rx.el.p(text(row, "active_title"), class_name="small muted"),
            ),
            chip("waiting for you", tone="orange"),
        ),
        rx.el.div(
            rx.cond(
                kind == "code_review",
                code_review(row),
                rx.cond(
                    kind == "validation_review",
                    validation_review(row),
                    already_clean_review(row),
                ),
            ),
            class_name="panel-body",
        ),
    )


def page() -> rx.Component:
    waiting = State.waiting_review_rows.to(list[dict[str, Any]])
    return view(
        "review",
        view_head(
            "Review",
            "The run pauses on each decision until you answer it. Every other task keeps running "
            "while it waits.",
        ),
        rx.cond(
            State.review_waiting_count > 0,
            rx.el.div(
                rx.el.div(rx.foreach(waiting, queue_entry), class_name="panel"),
                rx.el.div(
                    rx.cond(State.pending_user_validation, user_validation(), rx.fragment()),
                    rx.foreach(waiting, question),
                    style={"display": "flex", "flexDirection": "column", "gap": "16px"},
                ),
                class_name="review-layout",
            ),
            panel(
                rx.el.div(
                    rx.el.div(rx.el.span("✓"), class_name="ic"),
                    rx.el.h2("No decisions waiting"),
                    rx.el.p(
                        rx.cond(
                            State.is_cleaning,
                            "The run is working. It will stop here if it needs you.",
                            "Turn on human-in-the-loop, or set the validator to USER, to be asked.",
                        )
                    ),
                    class_name="empty",
                )
            ),
        ),
    )
