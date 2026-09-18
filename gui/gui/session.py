"""One running pipeline per browser session.

A cleaning run lives in a worker thread, and the page talks to it through a stop flag and two
queues of questions waiting for an answer. Those used to be module-level globals, so a second tab
shared the first tab's run: its stop button stopped that run, and its review queue showed that
run's questions. Everything here is keyed by the session's own token instead, so two tabs are two
runs that cannot see each other.

The conditions stay threading primitives rather than anything asyncio: the pipeline's callbacks are
called from the worker thread, which blocks on them until the page answers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SessionRun:
    """The state one browser session's run shares with its worker thread."""

    # Set when the person presses stop. The pipeline checks it between tasks.
    cancel: threading.Event = field(default_factory=threading.Event)

    # Questions the validator is waiting on, and the answers the page has given.
    validation_condition: threading.Condition = field(default_factory=threading.Condition)
    validation_pending: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    validation_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # The same, for human-in-the-loop code and recommendation reviews.
    hitl_condition: threading.Condition = field(default_factory=threading.Condition)
    hitl_pending: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hitl_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def start(self) -> None:
        """Clears the last run's leftovers so a new run starts with nothing waiting."""
        self.cancel = threading.Event()
        with self.validation_condition:
            self.validation_pending.clear()
            self.validation_results.clear()
        with self.hitl_condition:
            self.hitl_pending.clear()
            self.hitl_results.clear()

    def is_cancelled(self) -> bool:
        return self.cancel.is_set()


_RUNS: Dict[str, SessionRun] = {}
_RUNS_LOCK = threading.Lock()


def session_run(token: str) -> SessionRun:
    """The run belonging to this session, created the first time it is asked for.

    An empty token would put every session that has not identified itself into the same run, which
    is the bug this module exists to fix, so it is kept apart under its own name instead.
    """
    key = token or "unknown-session"
    with _RUNS_LOCK:
        run = _RUNS.get(key)
        if run is None:
            run = SessionRun()
            _RUNS[key] = run
        return run


def forget_session(token: str) -> None:
    """Drops a session's run. Used when a session ends, and by the tests."""
    with _RUNS_LOCK:
        _RUNS.pop(token or "unknown-session", None)


def live_sessions() -> int:
    """How many sessions currently hold a run. For the tests and for debugging."""
    with _RUNS_LOCK:
        return len(_RUNS)
