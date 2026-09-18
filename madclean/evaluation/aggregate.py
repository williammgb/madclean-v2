"""Averaging the scores of several runs.

A single run of an LLM system says little: the same dataset scores differently each time. The
paper reports the average of four runs, so this module averages the same way and adds what the
average hides — how far the runs were apart.

The mean is the plain arithmetic mean, so it reproduces the committed averages exactly. The spread
is the population standard deviation, which is 0.0 for one run rather than undefined.
"""

from __future__ import annotations

import math
from dataclasses import asdict, fields

from madclean.evaluation.scores import AggregateScores, Scores, Spread


def aggregate(runs: list[Scores]) -> AggregateScores:
    """Averages a list of run scores field by field."""
    if not runs:
        raise ValueError("Averaging needs at least one run.")
    sections = {}
    for section in fields(Scores):
        values_per_field: dict[str, list[float]] = {}
        for run in runs:
            for name, value in asdict(getattr(run, section.name)).items():
                values_per_field.setdefault(name, []).append(float(value))
        sections[section.name] = {name: spread(values) for name, values in values_per_field.items()}
    return AggregateScores(**sections)


def aggregate_per_column(runs: list[dict[str, Scores]]) -> dict[str, AggregateScores]:
    """Averages per-column scores over runs, for the columns every run scored."""
    if not runs:
        raise ValueError("Averaging needs at least one run.")
    shared = set(runs[0])
    for run in runs[1:]:
        shared &= set(run)
    return {column: aggregate([run[column] for run in runs]) for column in sorted(shared)}


def spread(values: list[float]) -> Spread:
    """The mean and the population standard deviation of one measurement."""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return Spread(mean=mean, standard_deviation=math.sqrt(variance), runs=len(values))


def means_only(scores: AggregateScores) -> dict:
    """The averages in the shape the committed results JSON stores them, without the spread.

    The stored `avg_eval_results.json` files hold one number per field. Keeping a function that
    produces exactly that shape is what lets the tests compare against them.
    """
    return {
        section.name: {
            name: value.mean for name, value in getattr(scores, section.name).items()
        }
        for section in fields(AggregateScores)
    }
