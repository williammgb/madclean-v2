"""Keeps a run alive on a console that cannot encode the characters MADClean prints.

Progress lines name functional dependency tasks as "left -> right" with an arrow character, and
prompts and profiles carry other non-ASCII text. On Windows the standard streams default to
cp1252, so the first such character raises UnicodeEncodeError inside print() and takes the whole
cleaning run down with it. Replacing the unencodable characters loses a glyph in the terminal;
crashing loses the run.
"""

import sys


def configure_console() -> None:
    """Make stdout and stderr replace characters they cannot encode instead of raising."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass
