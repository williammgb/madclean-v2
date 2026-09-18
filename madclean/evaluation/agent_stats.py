"""Reading a run report as a record of how hard the agents worked.

Scores say how well a run cleaned; these say what it cost to get there. Every number comes out of
the report the run already produced, so nothing here re-runs or re-counts anything: attempts and
validator rejections are read from the trace the agent loop kept, and the tokens from the per-task
counters the cleaner filled in as it went.
"""

from __future__ import annotations

from madclean.components.domain.report import (
    CleaningReport,
    ColumnReport,
    FDReport,
    SkippedColumnReport,
)
from madclean.evaluation.scores import AgentStats, ColumnAgentStats

# A validator step whose status is this one sent the work back for another attempt.
REJECTED = "needs_correction"


def agent_stats(report: CleaningReport) -> AgentStats:
    """Turns one run's report into per-column and whole-run agent statistics."""
    stats = AgentStats(runtime_seconds=report.runtime_seconds)
    for key, entry in report.entries.items():
        usage = report.per_task_usage.get(key)
        stats.per_column[key] = ColumnAgentStats(
            attempts=entry.attempts,
            validator_rejections=_rejections(entry),
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            already_clean=getattr(entry, "already_clean", False),
            cleaned=entry.cleaned,
            validated=entry.cleaning_validated,
            failed=_failed(entry),
        )
    return stats


def _rejections(entry: ColumnReport | FDReport | SkippedColumnReport) -> int:
    """How often the validator asked this column's work to be done again."""
    # Column steps are "validator_1", dependency steps "fd_validator_1", so match on the word.
    steps = getattr(entry, "trace_steps", [])
    return sum(1 for step in steps if "validator" in step.id and step.status == REJECTED)


def _failed(entry: ColumnReport | FDReport | SkippedColumnReport) -> bool:
    """A task that never produced a cleaned column and says why it did not."""
    if isinstance(entry, SkippedColumnReport):
        return False
    return not entry.cleaned and entry.reason is not None
