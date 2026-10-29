# F6 · Training in practice

Not started yet.

When it starts, set it up like every project here, in its own virtual environment:

```bash
./setup_venv.sh f6-training
source f6-training/.venv/bin/activate
```

**The exercise:** Take apart the F1 training loop one piece at a time: batches, cross-entropy loss, SGD versus AdamW, learning-rate warmup and decay, initialization, gradient clipping, and the validation split. Change one thing per run and plot what it does to the loss.

Written up at [danielrushton.com/writing/f6-training-in-practice](https://danielrushton.com/writing/f6-training-in-practice/) once published.
