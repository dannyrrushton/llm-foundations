# llm-foundations: conventions

- One folder per project (`fN-<name>/`, numbered to match the post: F1, F2, ...), each self-contained: its own `README.md`,
  `requirements.txt` and scripts.
- Every project runs in its own Python virtual environment at `<folder>/.venv`, created with
  `./setup_venv.sh <folder>`. Never install packages into the system Python or with `--user`, and
  never use `--break-system-packages`. READMEs give setup as:

  ```bash
  ./setup_venv.sh <folder>
  source <folder>/.venv/bin/activate
  ```

- Scripts warn when run outside a virtual environment (see `venv_check.py`, imported first by each
  project's shared module).
- `requirements.txt` lists every package a project imports, including optional extras that libraries
  only ask for at run time (for example datatrove's `orjson` and `spacy`).
- Console colour comes from `console.py` at the repo root (no dependencies), imported as
  `from console import ...` after `common`. Use its semantic helpers rather than raw escape codes:
  `heading`, `dim`, `good`/`bad`, `warn`, `path`, `generated`, `check(err, tol)` for by-hand-vs-library
  errors, and `raise fail(msg)` for exits. Pad a field before colouring it, and keep piped output
  identical to uncoloured output.
- `.venv/`, `data/` and `out/` are git-ignored.
