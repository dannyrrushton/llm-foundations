# llm-foundations

The foundations prelude to the series at [danielrushton.com](https://danielrushton.com/writing/): one
idea per project, each built by hand and then checked against a real library. Self-paced, so posts go up
as they're finished. Each project has its own folder and its own post.

| Project | Folder | Status |
| --- | --- | --- |
| F1 · The whole pipeline on one page | [`f1-mini-pipeline`](f1-mini-pipeline/) | Code ready |
| F2 · Gradient descent by hand | [`f2-gradient-descent`](f2-gradient-descent/) | Planned |
| F3 · Backpropagation by hand | [`f3-backprop`](f3-backprop/) | Planned |
| F4 · Attention by hand | [`f4-attention`](f4-attention/) | Planned |
| F5 · Training in practice | [`f5-training`](f5-training/) | Planned |
| F6 · Training a tiny GPT on one GPU | [`f6-nanogpt`](f6-nanogpt/) | Planned |
| F7 · How big is the KV cache? | [`f7-kv-cache`](f7-kv-cache/) | Planned |
| F8 · Bootstrapping a benchmark | [`f8-eval-ci`](f8-eval-ci/) | Planned |

Each folder has its own README, `requirements.txt` and instructions.

## Setup

Each project runs in its own Python virtual environment, so packages never clash between projects or
with your system Python. From the repo root:

```bash
./setup_venv.sh f1-mini-pipeline
source f1-mini-pipeline/.venv/bin/activate
```

`setup_venv.sh` creates `<folder>/.venv` if it's missing and installs that folder's `requirements.txt`.
Run it again after the requirements change. It needs Python 3.10 or later; set `PYTHON=python3.12` (or
similar) to pick the interpreter. The scripts print a warning if they're run outside a virtual
environment.
