"""A cp1252 console must not kill a run.

The first live benchmark run died at the functional dependency stage: the progress line for the
FD task "codex -> cityx" contains an arrow character, and Windows gives a piped process a cp1252
stdout, where printing it raises UnicodeEncodeError. Every column had already been cleaned and
paid for when the run crashed, so this is worth a test that bites.
"""

import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]

# The real progress line, as cleaning_coordinator._update_progress prints it for an FD task.
FD_LINE = "[codex → cityx] Successfully cleaned."


def _run_under_cp1252(body: str) -> subprocess.CompletedProcess:
    """Runs a snippet in a child process whose stdout encodes as cp1252, like a piped run."""
    script = textwrap.dedent(
        f"""
        import sys
        sys.stdout.reconfigure(encoding="cp1252")
        {body}
        print({FD_LINE!r})
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=CODE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_a_cp1252_console_crashes_without_the_fix():
    """Without configure_console the arrow is fatal — otherwise the test below proves nothing."""
    result = _run_under_cp1252("pass")
    assert result.returncode != 0
    assert "UnicodeEncodeError" in result.stderr


def test_configure_console_keeps_an_fd_progress_line_printable():
    result = _run_under_cp1252("from madclean.utils.console import configure_console; configure_console()")
    assert result.returncode == 0, result.stderr
    assert "Successfully cleaned." in result.stdout
    assert "codex" in result.stdout and "cityx" in result.stdout


def test_every_console_entry_point_configures_the_console():
    """The helper only helps where it is called: the CLI, the UI launcher and the live runner."""
    from evaluation import live_run
    from madclean import main

    for function in (main.cli, main.cli_ui, live_run.main):
        source = inspect.getsource(function)
        assert "configure_console()" in source, f"{function.__qualname__} does not configure the console"
