"""The Guide view: what this system does, in the order a person meets it."""

from __future__ import annotations

import reflex as rx

from ..components.shell import panel, panel_head, view, view_head

STEPS: list[tuple[str, str]] = [
    (
        "Upload a dataset",
        "A CSV or an Excel file. Nothing leaves this machine at this point; the table is read "
        "locally and shown as it is.",
    ),
    (
        "Look at the profile",
        "Profiling runs here, not in a model: it names each column's semantic type, finds "
        "outliers, and looks for dependencies between columns. The Table view's header colours "
        "come from it.",
    ),
    (
        "Choose the models",
        "One model per agent. The Recommender decides what a column needs, the Coder writes the "
        "cleaning code, and the Validator checks the result. Setting the Validator to USER puts "
        "you in that seat.",
    ),
    (
        "Run the cleaning",
        "Columns are cleaned at the same time, up to fifteen at once. A dependency rule waits "
        "until both of its columns are done. The Pipeline view shows each task step by step.",
    ),
    (
        "Answer what it asks",
        "With human-in-the-loop on, the run pauses on each decision and waits for you. The rest "
        "of the run keeps going while it waits.",
    ),
    (
        "Read the result",
        "The Report has the tokens, the runtime and the code the agents wrote. The Evaluation "
        "view scores the run against a ground-truth file if you have one.",
    ),
]

TERMS: list[tuple[str, str]] = [
    ("Semantic type", "What a column means, not how it is stored: a date, a city, a rating."),
    (
        "Functional dependency",
        "A pair of columns where one decides the other, like a brewery id deciding its state. "
        "MADClean uses them to find rows that contradict each other.",
    ),
    ("Outlier", "A number far enough from the rest of its column to be worth a second look."),
    (
        "Validator",
        "The agent that reads the cleaned column against the original and either accepts it or "
        "sends it back with instructions.",
    ),
]


def page() -> rx.Component:
    return view(
        "guide",
        view_head("Guide"),
        rx.el.div(
            panel(
                panel_head(rx.el.h2("Working through a dataset")),
                rx.el.ol(
                    *[
                        rx.el.li(rx.el.div(rx.el.b(title), rx.el.p(text)))
                        for title, text in STEPS
                    ],
                    class_name="steps-list",
                ),
            ),
            panel(
                panel_head(rx.el.h2("Words used here")),
                rx.el.div(
                    rx.el.table(
                        rx.el.tbody(
                            *[
                                rx.el.tr(rx.el.td(rx.el.b(term)), rx.el.td(meaning))
                                for term, meaning in TERMS
                            ]
                        ),
                        class_name="data",
                    ),
                    class_name="scroll-x",
                ),
            ),
            class_name="guide",
        ),
    )
