"""Colour for the scripts' terminal output, with no dependencies.

Colour is on only when the output goes to a terminal, so piped or redirected output stays plain
text. Set NO_COLOR=1 to turn it off (https://no-color.org) or FORCE_COLOR=1 to keep it on when piped.

Pad a field before colouring it, since the escape codes count towards f-string widths:
    print(f"{good(f'{name:<11}')} {loss:.3f}")
"""

import os
import sys

if os.name == "nt":
    os.system("")  # lets Windows 10+ consoles interpret ANSI escape codes

_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "magenta": "35", "cyan": "36"}


def color_enabled(stream=None) -> bool:
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return os.environ.get("TERM") != "dumb" and hasattr(stream, "isatty") and stream.isatty()


def style(text, *names, stream=None) -> str:
    """Wrap text in the named styles, e.g. style("done", "bold", "green")."""
    if text == "" or not color_enabled(stream):
        return str(text)
    return f"\033[{';'.join(_CODES[n] for n in names)}m{text}\033[0m"


# What each colour means, used the same way in every project.
def heading(t, stream=None):   return style(t, "bold", stream=stream)        # section titles, table headers
def dim(t, stream=None):       return style(t, "dim", stream=stream)         # labels, secondary detail
def good(t, stream=None):      return style(t, "green", stream=stream)       # passed checks, kept, ok
def bad(t, stream=None):       return style(t, "red", stream=stream)         # failed checks, errors
def warn(t, stream=None):      return style(t, "yellow", stream=stream)      # worth a look, not fatal
def path(t, stream=None):      return style(t, "cyan", stream=stream)        # files written or read
def generated(t, stream=None): return style(t, "magenta", stream=stream)     # text the model generated


def check(err, tol, fmt=".2e", width=0) -> str:
    """Format a by-hand-vs-library error, right-aligned to width: green within tol, red beyond it."""
    s = format(err, fmt).rjust(width)
    return good(s) if err <= tol else bad(s)


def fail(message) -> SystemExit:
    """An exit with the message in red on stderr: raise fail("No data. Run fetch.py first.")"""
    return SystemExit(bad(message, stream=sys.stderr))
