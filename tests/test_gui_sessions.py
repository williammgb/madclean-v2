"""Two browser tabs are two runs.

The stop flag and the two queues of questions waiting for an answer used to be module-level
globals in the GUI's state, so a second tab stopped the first tab's run and saw its review
questions. They are keyed by session now, and these tests hold both sessions at once to show it.
"""

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))

from gui.session import SessionRun, forget_session, live_sessions, session_run  # noqa: E402


@pytest.fixture(autouse=True)
def clean_registry():
    for token in ("tab-one", "tab-two", "", "unknown-session"):
        forget_session(token)
    yield
    for token in ("tab-one", "tab-two", "", "unknown-session"):
        forget_session(token)


def test_each_session_gets_its_own_run():
    first = session_run("tab-one")
    second = session_run("tab-two")

    assert first is not second
    assert session_run("tab-one") is first, "the same tab must keep the same run"
    assert live_sessions() >= 2


def test_stopping_one_session_leaves_the_other_running():
    first = session_run("tab-one")
    second = session_run("tab-two")

    first.cancel.set()

    assert first.is_cancelled() is True
    assert second.is_cancelled() is False


def test_a_question_waits_in_its_own_session_only():
    first = session_run("tab-one")
    second = session_run("tab-two")

    with first.validation_condition:
        first.validation_pending["req-1"] = {"column": "city"}
    with second.hitl_condition:
        second.hitl_pending["req-2"] = {"column": "abv"}

    assert list(first.validation_pending) == ["req-1"]
    assert first.hitl_pending == {}
    assert list(second.hitl_pending) == ["req-2"]
    assert second.validation_pending == {}


def test_a_new_run_clears_only_that_session():
    first = session_run("tab-one")
    second = session_run("tab-two")
    first.validation_pending["old"] = {}
    second.validation_pending["kept"] = {}
    first.cancel.set()

    first.start()

    assert first.validation_pending == {}
    assert first.is_cancelled() is False
    assert list(second.validation_pending) == ["kept"], "the other tab's run was cleared too"


def test_a_session_without_a_token_does_not_join_another_session():
    """A session that never identified itself gets its own run, not everyone else's."""
    anonymous = session_run("")
    named = session_run("tab-one")

    assert anonymous is not named
    assert session_run("") is anonymous


def test_the_worker_thread_and_the_page_meet_through_one_session():
    """What the run's thread waits for is what that session's page answers, and nothing else's."""
    answering = session_run("tab-one")
    other = session_run("tab-two")
    answers: dict[str, dict] = {}

    def worker():
        # What the pipeline's callback does: queue the question, then block until it is answered.
        with answering.validation_condition:
            answering.validation_pending["req"] = {"column": "city"}
        deadline = time.time() + 5
        while time.time() < deadline:
            with answering.validation_condition:
                if "req" in answering.validation_results:
                    answers["result"] = dict(answering.validation_results.pop("req"))
                    return
            time.sleep(0.01)

    thread = threading.Thread(target=worker)
    thread.start()

    # The other tab answers first; it must not reach the waiting thread.
    deadline = time.time() + 5
    while "req" not in answering.validation_pending and time.time() < deadline:
        time.sleep(0.01)
    with other.validation_condition:
        other.validation_results["req"] = {"needs_correction": True}
    time.sleep(0.1)
    assert answers == {}, "another tab's answer reached this run"

    with answering.validation_condition:
        answering.validation_results["req"] = {"needs_correction": False}
        answering.validation_condition.notify_all()
    thread.join(timeout=5)

    assert answers["result"] == {"needs_correction": False}


def test_a_run_is_a_plain_object_the_thread_can_hold():
    """The worker thread is handed the run itself, so it never has to reach into the page."""
    run = session_run("tab-one")
    assert isinstance(run, SessionRun)
    assert callable(run.is_cancelled)
