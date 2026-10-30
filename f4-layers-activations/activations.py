"""Every common activation function, written by hand in NumPy, then checked against PyTorch.

For each function: the value and its derivative, computed from the formula, compared with
torch.nn.functional (values) and torch.autograd (derivatives) on 10,001 points from -6 to 6.
Then a table of the properties that decide where each one gets used, a gallery of all of them,
and a separate x/y plot of each one.

Output: out/activations.png (the gallery), out/activations/<name>.png (one plot per function), and
the comparison table printed to the terminal.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

import common  # noqa: F401  (venv check)
from common import OUT
from console import check, dim, heading, path  # common.py puts the repo root on sys.path

TOL = 1e-9  # float64 on both sides: a correct formula agrees to ~1e-15

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


# For the one-plot-per-function figures: the formula as a title, and a y range that shows the shape.
FORMULAS = {
    "sigmoid": r"$\sigma(x) = \dfrac{1}{1 + e^{-x}}$",
    "tanh": r"$\tanh(x) = \dfrac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$",
    "relu": r"$\mathrm{ReLU}(x) = \max(0,\ x)$",
    "leaky_relu": r"$\mathrm{LeakyReLU}(x) = \max(0.01x,\ x)$",
    "gelu": r"$\mathrm{GELU}(x) = x\,\Phi(x)$",
    "gelu_tanh": r"$\mathrm{GELU}(x) \approx \frac{x}{2}\left(1 + \tanh\left(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\right)\right)$",
    "silu": r"$\mathrm{SiLU}(x) = x\,\sigma(x)$",
}
YLIM = {"sigmoid": (-0.25, 1.25), "tanh": (-1.35, 1.35)}  # the rest share (-1, 4)
ASYMPTOTES = {"sigmoid": (0, 1), "tanh": (-1, 1)}


def plot_each(plt, folder):
    """One x/y plot per activation: f(x) solid, f'(x) dashed, asymptotes and minima marked."""
    folder.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(-5, 5, 1001)
    written = []
    for name, (f, df) in FUNCS.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.grid(True, color="0.92", lw=0.8)
        ax.axhline(0, color="0.55", lw=0.9)
        ax.axvline(0, color="0.55", lw=0.9)
        for y in ASYMPTOTES.get(name, ()):
            ax.axhline(y, color="0.6", lw=0.9, ls=":")
        ax.plot(xs, f(xs), lw=2.4, color="C0", label="f(x)")
        ax.plot(xs, df(xs), lw=1.6, ls="--", color="C1", label="f'(x)")
        ys = f(xs)
        if ys.min() < -1e-3 and name not in ("tanh", "leaky_relu"):  # GELU's and SiLU's dip below zero
            i = ys.argmin()
            ax.plot(xs[i], ys[i], "o", color="C0", ms=5)
            ax.annotate(f"min {ys[i]:.3f} at x = {xs[i]:.2f}", (xs[i], ys[i]), xytext=(-14, 46), ha="right",
                        textcoords="offset points", fontsize=9, arrowprops=dict(arrowstyle="-", color="0.5"))
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(*YLIM.get(name, (-1, 4)))
        ax.set_xlabel("x  (the neuron's input)")
        ax.set_ylabel("y")
        ax.set_title(f"{name}\n{FORMULAS[name]}", fontsize=11)
        ax.legend(loc="upper left", frameon=False)
        rng, centred, flat, use = PROPS[name]
        ax.text(0.99, 0.02, f"range {rng} · flat where: {flat}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="0.4")
        fig.tight_layout()
        fig.savefig(folder / f"{name}.png", dpi=130)
        plt.close(fig)
        written.append(name)
    return written


def softmax(z):
    """The output layer's 'activation': scores -> probabilities. Subtract the max first so exp can't overflow."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    x = np.linspace(-6, 6, 10_001)
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)

    print(heading(f"{'function':<11} {'max |f - torch|':>16} {'max |df - autograd|':>20}   {'range':<13} {'0-centred':<9} {'flat where':<14} used in"))
    for name, (f, df) in FUNCS.items():
        y_t = TORCH[name](xt)
        (g_t,) = torch.autograd.grad(y_t.sum(), xt)
        err_f = np.abs(f(x) - y_t.detach().numpy()).max()
        # ReLU's derivative at exactly 0 is a convention (PyTorch uses 0); skip that one point.
        mask = x != 0
        err_df = np.abs(df(x)[mask] - g_t.numpy()[mask]).max()
        rng, centred, flat, use = PROPS[name]
        print(f"{heading(f'{name:<11}')} {check(err_f, TOL, width=16)} {check(err_df, TOL, width=20)}   "
              f"{rng:<13} {centred:<9} {flat:<14} {dim(use)}")

    # Softmax: the last step of the network, scores for every token -> a probability distribution.
    z = np.random.default_rng(0).normal(size=(4, 4096)) * 3
    err = np.abs(softmax(z) - torch.softmax(torch.tensor(z), -1).numpy()).max()
    print(f"\nsoftmax over 4,096 scores: max |by hand - torch| = {check(err, TOL)}; rows sum to {softmax(z).sum(-1).round(12)}")
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
    plt.close(fig)
    print(f"\nwrote {path(OUT / 'activations.png')} ({len(names)} functions)")

    each = plot_each(plt, OUT / "activations")
    print(f"wrote {path(OUT / 'activations' / '<name>.png')} (one x/y plot each: {', '.join(each)})")


if __name__ == "__main__":
    main()
