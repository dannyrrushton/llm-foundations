# llm-foundations

Four weeks of reading and one exercise per week, before the main series at
[danielrushton.com](https://danielrushton.com/writing/). Each week has its own folder and its own post.

| Week | Folder | Post | Status |
| --- | --- | --- | --- |
| 1 · The big picture | [`week-1-mini-pipeline`](week-1-mini-pipeline/) | F1 · The whole pipeline on one page | Code ready |
| 2 · The transformer by hand | [`week-2-nanogpt`](week-2-nanogpt/) | F2 · Training a tiny GPT on one GPU | Planned |
| 3 · Training and inference | [`week-3-kv-cache`](week-3-kv-cache/) | F3 · How big is the KV cache? | Planned |
| 4 · Evaluating LLMs | [`week-4-eval-ci`](week-4-eval-ci/) | F4 · Bootstrapping a benchmark | Planned |

Each folder has its own README, `requirements.txt` and instructions.

## Setup

Each project runs in its own Python virtual environment, so packages never clash between projects or
with your system Python. From the repo root:

```bash
./setup_venv.sh week-1-mini-pipeline
source week-1-mini-pipeline/.venv/bin/activate
```

`setup_venv.sh` creates `<folder>/.venv` if it's missing and installs that folder's `requirements.txt`.
Run it again after the requirements change. It needs Python 3.10 or later; set `PYTHON=python3.12` (or
similar) to pick the interpreter. The scripts print a warning if they're run outside a virtual
environment.
