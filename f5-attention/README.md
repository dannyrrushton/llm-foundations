# F5 · Attention by hand

Not started yet.

When it starts, set it up like every project here, in its own virtual environment:

```bash
./setup_venv.sh f5-attention
source f5-attention/.venv/bin/activate
```

**The exercise:** Write one causal self-attention head from matrix multiplies, a mask and a softmax, then extend it to several heads. Check the output against PyTorch's scaled_dot_product_attention to within about 1e-6, and plot the attention weights for a sentence.

Written up at [danielrushton.com/writing/f5-attention-by-hand](https://danielrushton.com/writing/f5-attention-by-hand/) once published.
