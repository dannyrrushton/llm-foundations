# F9 · Bootstrapping a benchmark

Not started yet.

When it starts, set it up like every project here, in its own virtual environment:

```bash
./setup_venv.sh f9-eval-ci
source f9-eval-ci/.venv/bin/activate
```

**The exercise:** Run one small benchmark on a local model with lm-evaluation-harness, then compute a 95% bootstrap confidence interval by hand and compare it with the harness's standard error.

Written up at [danielrushton.com/writing/f9-bootstrapping-a-benchmark](https://danielrushton.com/writing/f9-bootstrapping-a-benchmark/) once published.
