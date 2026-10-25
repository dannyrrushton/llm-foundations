# F7 · How big is the KV cache?

Not started yet.

When it starts, set it up like every project here, in its own virtual environment:

```bash
./setup_venv.sh f7-kv-cache
source f7-kv-cache/.venv/bin/activate
```

**The exercise:** Work out by hand how much KV cache a 7B model needs for one 4,096-token request, then check it against what vLLM reports when serving the same model.

Written up at [danielrushton.com/writing/f7-kv-cache-math](https://danielrushton.com/writing/f7-kv-cache-math/) once published.
