# F4 · Inside the network: layers and activation functions

The tiny GPT from F1, taken apart layer by layer. What each layer does to the data, where the
parameters live, why the network needs activation functions at all, what the common ones look like
and why GELU (and now SwiGLU) won out for transformers.

Written up at [danielrushton.com/writing/f4-layers-and-activations](https://danielrushton.com/writing/f4-layers-and-activations/) once published.

| Script | What it does | Output |
| --- | --- | --- |
| `why_nonlinearity.py` | Shows that stacked linear layers collapse into one, then fits `sin(3x)` with and without activations | `out/why_nonlinearity.png` |
| `activations.py` | Sigmoid, tanh, ReLU, Leaky ReLU, GELU (both forms), SiLU and softmax written by hand in NumPy; values checked against `torch.nn.functional`, derivatives against autograd; a gallery of all of them plus one labelled x/y plot per function (value, derivative, asymptotes, minimum) | `out/activations.png`, `out/activations/<name>.png` |
| `inspect_layers.py` | Every layer of the tiny GPT with its shape and parameter count; one forward pass written out by hand and checked against the model; what GELU actually sees in each block; the top next-token predictions | `out/mlp_preactivations.png` |
| `swap_activation.py` | Trains the same tiny GPT once per MLP activation (none, sigmoid, tanh, ReLU, GELU, SiLU, SwiGLU) and compares validation loss, dead units and speed | `out/swap_activation.png`, `.json` |

The model in `common.py` is F1's, with the same layer names, so F1's trained checkpoint loads
unchanged. The only addition is a switch for the MLP's activation function.

## Run it

Set up this project's own virtual environment from the repo root, then work inside it:

```bash
./setup_venv.sh f4-layers-activations
source f4-layers-activations/.venv/bin/activate
cd f4-layers-activations
```

```bash
python why_nonlinearity.py     # seconds, CPU is fine
python activations.py          # seconds
python inspect_layers.py       # seconds; uses F1's out/base.pt and tokenizer if they exist
python swap_activation.py      # 7 short training runs: a few minutes on a GPU, longer on CPU
```

`inspect_layers.py` and `swap_activation.py` work best after running F1 through `pretrain.py`
(for `inspect_layers.py`) or `train_tokenizer.py` (for `swap_activation.py`). Without F1,
`inspect_layers.py` uses a freshly initialized model, and `swap_activation.py` can train on
characters from any text file:

```bash
python swap_activation.py --text some-book.txt
python swap_activation.py --activations relu gelu swiglu --steps 2000
```
