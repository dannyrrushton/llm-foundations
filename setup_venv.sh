#!/usr/bin/env bash
# Create (or update) a project's own Python virtual environment and install its requirements.
#
#   ./setup_venv.sh f1-mini-pipeline
#   source f1-mini-pipeline/.venv/bin/activate
#
# Every project folder gets its own .venv, so one project's packages (torch, vLLM, ...) never clash
# with another's or with the system Python. Run it again after changing requirements.txt.
# Set PYTHON to choose the interpreter, e.g. PYTHON=python3.12 ./setup_venv.sh f1-mini-pipeline
set -euo pipefail

if [ $# -ne 1 ] || [ ! -d "$1" ]; then
  echo "usage: $0 <project-folder>    e.g. $0 f1-mini-pipeline" >&2
  exit 1
fi
project="${1%/}"
req="$project/requirements.txt"
[ -f "$req" ] || { echo "No $req: nothing to install." >&2; exit 1; }

python="${PYTHON:-python3}"
"$python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || { echo "Python 3.10 or later is needed; $python is $("$python" -V 2>&1). Set PYTHON=..." >&2; exit 1; }

if [ ! -d "$project/.venv" ]; then
  echo "Creating $project/.venv with $("$python" -V 2>&1)"
  "$python" -m venv "$project/.venv"
fi

"$project/.venv/bin/python" -m pip install --upgrade pip --quiet
"$project/.venv/bin/python" -m pip install -r "$req"

echo
echo "Done. Activate it with:"
echo "  source $project/.venv/bin/activate"
