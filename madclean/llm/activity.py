"""A live record of which model calls are open right now, so a screen can show the run is still talking.

Each Pipeline owns one record: the GUI runs several sessions at once, so it cannot be global. Who is
calling travels in ContextVars, which every asyncio task copies, so columns cleaned at once never
mix up their agent, task or attempt.
"""

import itertools
import threading
import time
from contextvars import ContextVar

# Which agent, which column or dependency task, and which attempt of its loop the current call is for.
current_agent: ContextVar[str | None] = ContextVar("madclean_agent", default=None)
current_task: ContextVar[str | None] = ContextVar("madclean_task_key", default=None)
current_attempt: ContextVar[int | None] = ContextVar("madclean_attempt", default=None)


class LLMActivity:
    """The model calls open right now, and when the last reply came. Safe to use from any thread."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._open: dict[int, dict] = {}
        self._last_reply_at: float | None = None

    def start(self, agent, task, attempt, model) -> int:
        """Record a call that has just been sent; returns the id that `retrying` and `finish` take."""
        with self._lock:
            call_id = next(self._ids)
            self._open[call_id] = {
                "agent": agent,
                "task": task,
                "attempt": attempt,
                "model": model,
                "started": self._clock(),
                "try_no": 1,
                "tries": None,
                "reason": None,
            }
            return call_id

    def retrying(self, call_id: int, try_no: int, tries: int, reason: str) -> None:
        """The call is being sent again: try `try_no` of `tries`, because it `reason` (timed out, ...)."""
        with self._lock:
            call = self._open.get(call_id)
            if call is not None:
                call.update(try_no=try_no, tries=tries, reason=reason)

    def finish(self, call_id: int) -> None:
        """The call is over, answered or failed; it is no longer open."""
        with self._lock:
            if self._open.pop(call_id, None) is not None:
                self._last_reply_at = self._clock()

    def snapshot(self) -> dict:
        """What is open now, oldest first, with how long each has run, and seconds since the last reply."""
        with self._lock:
            now = self._clock()
            open_calls = [
                {
                    "agent": call["agent"],
                    "task": call["task"],
                    "attempt": call["attempt"],
                    "model": call["model"],
                    "seconds": now - call["started"],
                    "try_no": call["try_no"],
                    "tries": call["tries"],
                    "reason": call["reason"],
                }
                for _, call in sorted(self._open.items())
            ]
            last = None if self._last_reply_at is None else now - self._last_reply_at
            return {"open": open_calls, "last_reply_seconds": last}
