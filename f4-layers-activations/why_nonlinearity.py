"""Why a network needs activation functions at all.

Part 1, the algebra: two linear layers with nothing between them are one linear layer. Multiply the
weight matrices together and the stack gives the same output, to float precision.

Part 2, the consequence: train the same small MLP (1 -> 64 -> 64 -> 1) to fit y = sin(3x), once with
no activation and once each with ReLU, tanh and GELU. Without one, the best it can do is a straight
line, however many layers it has.

Output: out/why_nonlinearity.png, and the final losses printed to the terminal.
"""

import torch
import torch.nn as nn

import common  # noqa: F401  (venv check)
from common import ACTIVATIONS, OUT
from console import check, dim, heading, path, warn  # common.py puts the repo root on sys.path


def part1():
    torch.manual_seed(0)
    l1, l2, l3 = nn.Linear(8, 32), nn.Linear(32, 32), nn.Linear(32, 4)
    x = torch.randn(1000, 8, dtype=torch.float64)
    l1, l2, l3 = l1.double(), l2.double(), l3.double()
    stacked = l3(l2(l1(x)))

    # Collapse: y = W3(W2(W1 x + b1) + b2) + b3 = (W3 W2 W1) x + (W3 W2 b1 + W3 b2 + b3)
    W = l3.weight @ l2.weight @ l1.weight
    b = l3.weight @ (l2.weight @ l1.bias + l2.bias) + l3.bias
    single = x @ W.T + b
    print(heading("Part 1: three linear layers, no activations"))
    print(f"  parameters in the stack: {sum(p.numel() for m in (l1, l2, l3) for p in m.parameters()):,}")
    print(f"  one equivalent layer:    {W.numel() + b.numel():,}  (W is {tuple(W.shape)})")
    print(f"  max |stack - single layer| = {check((stacked - single).abs().max().item(), 1e-9)}\n")  # float64

    with torch.no_grad():
        gelu_stack = l3(nn.functional.gelu(l2(nn.functional.gelu(l1(x)))))
    print(f"  with GELU between them, the single layer misses by up to {warn(f'{(gelu_stack - single).abs().max().item():.2f}')}:"
          " no matrix reproduces it.\n")


def part2(steps=3000):
    torch.manual_seed(0)
    x = torch.linspace(-2, 2, 400).unsqueeze(1)
    y = torch.sin(3 * x)
    results = {}
    print(heading("Part 2: fit y = sin(3x) with a 1 -> 64 -> 64 -> 1 MLP"))
    for name in ["none", "relu", "tanh", "gelu"]:
        torch.manual_seed(0)
        act = ACTIVATIONS[name]
        net = nn.Sequential(nn.Linear(1, 64), act(), nn.Linear(64, 64), act(), nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        for _ in range(steps):
            loss = ((net(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            results[name] = (net(x), loss.item())
        print(f"  {heading(f'{name:<5}')} final mean squared error {loss.item():.5f}")
    print(dim(f"  (a straight line's best possible error here is {line_error(x, y):.5f})"))
    return x, y, results


def line_error(x, y):
    A = torch.cat([x, torch.ones_like(x)], 1)
    coef = torch.linalg.lstsq(A, y).solution
    return ((A @ coef - y) ** 2).mean().item()


def main():
    part1()
    x, y, results = part2()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, color="black", lw=3, alpha=0.25, label="target: sin(3x)")
    for name, (pred, loss) in results.items():
        ax.plot(x, pred, lw=1.8, label=f"{name} (MSE {loss:.4f})")
    ax.legend()
    ax.set_title("Same network, different activation")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "why_nonlinearity.png", dpi=130)
    print(f"\nwrote {path(OUT / 'why_nonlinearity.png')}")


if __name__ == "__main__":
    main()
