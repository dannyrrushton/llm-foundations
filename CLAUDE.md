# llm-foundations: conventions

- One folder per project (`week-N-<name>/`), each self-contained: its own `README.md`,
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
- `.venv/`, `data/` and `out/` are git-ignored.
