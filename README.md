# llm-foundations

The foundations prelude to the series at [danielrushton.com](https://danielrushton.com/writing/): one
idea per project, each built by hand and then checked against a real library. Self-paced, so posts go up
as they're finished. Each project has its own folder and its own post.

| Project | Folder | Status |
| --- | --- | --- |
| F1 · The whole pipeline on one page | [`f1-mini-pipeline`](f1-mini-pipeline/) | Code ready |
| F2 · Inside the network: layers and activation functions | [`f2-layers-activations`](f2-layers-activations/) | Code ready |
| F3 · Gradient descent by hand | [`f3-gradient-descent`](f3-gradient-descent/) | Planned |
| F4 · Backpropagation by hand | [`f4-backprop`](f4-backprop/) | Planned |
| F5 · Attention by hand | [`f5-attention`](f5-attention/) | Planned |
| F6 · Training in practice | [`f6-training`](f6-training/) | Planned |
| F7 · Training a tiny GPT on one GPU | [`f7-nanogpt`](f7-nanogpt/) | Planned |
| F8 · How big is the KV cache? | [`f8-kv-cache`](f8-kv-cache/) | Planned |
| F9 · Bootstrapping a benchmark | [`f9-eval-ci`](f9-eval-ci/) | Planned |

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
