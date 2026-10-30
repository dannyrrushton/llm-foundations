"""Every common activation function, written by hand in NumPy, then checked against PyTorch.

For each function: the value and its derivative, computed from the formula, compared with
torch.nn.functional (values) and torch.autograd (derivatives) on 10,001 points from -6 to 6.
Then a table of the properties that decide where each one gets used, and a plot of all of them.

Output: out/activations.png, and the comparison table printed to the terminal.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

import common  # noqa: F401  (venv check)
from common import OUT

# ---------------------------------------------------------------------------
# The functions and their derivatives, by hand.
# ---------------------------------------------------------------------------

_erf = np.vectorize(math.erf)

# The Sigmoid function, which is the same as the logistic function. 
# Domain - any real number
# range (0, 1)
# asymptotes: y=0 as x -> -inf and y=1 as x -> inf
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def phi(x):  # standard normal CDF: the probability a N(0,1) sample is below x
    return 0.5 * (1 + _erf(x / math.sqrt(2)))


def pdf(x):  # standard normal density
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


C = math.sqrt(2 / math.pi)

FUNCS = {
    # name: (f(x), f'(x))
    "sigmoid": (sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x))),
    "tanh": (np.tanh, lambda x: 1 - np.tanh(x) ** 2),
    "relu": (lambda x: np.maximum(x, 0), lambda x: (x > 0).astype(float)),
    "leaky_relu": (lambda x: np.where(x > 0, x, 0.01 * x), lambda x: np.where(x > 0, 1.0, 0.01)),
    # GELU: x times the probability that x "survives" under N(0,1). Smooth ReLU, with a small dip below 0.
    "gelu": (lambda x: x * phi(x), lambda x: phi(x) + x * pdf(x)),
    # The tanh approximation GPT-2's code uses, from before erf was fast on GPUs.
    "gelu_tanh": (
        lambda x: 0.5 * x * (1 + np.tanh(C * (x + 0.044715 * x ** 3))),
        lambda x: 0.5 * (1 + np.tanh(C * (x + 0.044715 * x ** 3)))
        + 0.5 * x * (1 - np.tanh(C * (x + 0.044715 * x ** 3)) ** 2) * C * (1 + 3 * 0.044715 * x ** 2),
    ),
    # SiLU / Swish: x times its own sigmoid. The gate inside SwiGLU.
    "silu": (lambda x: x * sigmoid(x), lambda x: sigmoid(x) * (1 + x * (1 - sigmoid(x)))),
}

TORCH = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "relu": F.relu,
    "leaky_relu": lambda t: F.leaky_relu(t, 0.01),
    "gelu": F.gelu,
    "gelu_tanh": lambda t: F.gelu(t, approximate="tanh"),
    "silu": F.silu,
}

# Properties, from the formulas above. "Saturates" means the derivative goes to 0 for large |x|,
# which starves earlier layers of gradient (the vanishing-gradient problem).
PROPS = {
    "sigmoid": ("(0, 1)", "no", "both sides", "gates, probabilities; rarely hidden layers now"),
    "tanh": ("(-1, 1)", "yes", "both sides", "RNNs/LSTMs; inside GELU's approximation"),
    "relu": ("[0, inf)", "no", "x<0 is flat 0", "CNNs, early transformers; cheap"),
    "leaky_relu": ("(-inf, inf)", "no", "no", "ReLU without dead units"),
    "gelu": ("[-0.17, inf)", "no", "x<<0 only", "BERT, GPT-2/3, this tiny GPT"),
    "gelu_tanh": ("[-0.17, inf)", "no", "x<<0 only", "GPT-2's code; fast GELU"),
    "silu": ("[-0.28, inf)", "no", "x<<0 only", "the gate in SwiGLU (LLaMA, Mistral)"),
}


def softmax(z):
    """The output layer's 'activation': scores -> probabilities. Subtract the max first so exp can't overflow."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    x = np.linspace(-6, 6, 10_001)
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)

    print(f"{'function':<11} {'max |f - torch|':>16} {'max |df - autograd|':>20}   {'range':<13} {'0-centred':<9} {'flat where':<14} used in")
    for name, (f, df) in FUNCS.items():
        y_t = TORCH[name](xt)
        (g_t,) = torch.autograd.grad(y_t.sum(), xt)
        err_f = np.abs(f(x) - y_t.detach().numpy()).max()
        # ReLU's derivative at exactly 0 is a convention (PyTorch uses 0); skip that one point.
        mask = x != 0
        err_df = np.abs(df(x)[mask] - g_t.numpy()[mask]).max()
        rng, centred, flat, use = PROPS[name]
        print(f"{name:<11} {err_f:>16.2e} {err_df:>20.2e}   {rng:<13} {centred:<9} {flat:<14} {use}")

    # Softmax: the last step of the network, scores for every token -> a probability distribution.
    z = np.random.default_rng(0).normal(size=(4, 4096)) * 3
    err = np.abs(softmax(z) - torch.softmax(torch.tensor(z), -1).numpy()).max()
    print(f"\nsoftmax over 4,096 scores: max |by hand - torch| = {err:.2e}; rows sum to {softmax(z).sum(-1).round(12)}")
    big = np.array([1000.0, 999.0, 0.0])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(big) / np.exp(big).sum()
    print(f"why subtract the max: naive softmax of {big.tolist()} = {naive}, stable = {softmax(big).round(4)}")

    # The curves gallery (gelu_tanh is indistinguishable from gelu at this scale, so it's left off).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in FUNCS if n != "gelu_tanh"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    xs = np.linspace(-4, 4, 801)
    for ax, name in zip(axes.flat, ["sigmoid", "tanh", "relu", "leaky_relu", "gelu", "silu"]):
        f, df = FUNCS[name]
        ax.axhline(0, color="0.8", lw=0.8)
        ax.axvline(0, color="0.8", lw=0.8)
        ax.plot(xs, f(xs), lw=2.2, label="f(x)")
        ax.plot(xs, df(xs), lw=1.5, ls="--", label="f'(x)")
        ax.set_title(name)
        ax.set_ylim(-1.5, 4)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle("Activation functions (solid) and their derivatives (dashed)")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "activations.png", dpi=130)
    print(f"\nwrote {OUT / 'activations.png'} ({len(names)} functions)")


if __name__ == "__main__":
    main()
