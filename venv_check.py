"""Warn when a script runs outside a virtual environment.

Projects in this repo import this first, so running a script with the system Python prints a
reminder instead of failing later with a missing package or a clash with system packages.
"""

import os
import sys

from console import warn


def in_virtualenv() -> bool:
    return sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX"))


def warn_if_not_in_venv(project: str) -> None:
    if not in_virtualenv() and not os.environ.get("LLMF_SKIP_VENV_CHECK"):
        print(
            warn("Warning: not running in a virtual environment.", stream=sys.stderr)
            + f" Set one up from the repo root with:\n"
            f"  ./setup_venv.sh {project}\n"
            f"  source {project}/.venv/bin/activate\n"
            f"(Set LLMF_SKIP_VENV_CHECK=1 to silence this.)\n",
            file=sys.stderr,
        )
