"""Train the same tiny GPT several times, changing only the MLP's activation function.

Same data, same seed, same initialization scheme, same number of steps; only the non-linearity in
the MLP changes: none (the identity), sigmoid, tanh, ReLU, GELU, SiLU and SwiGLU. For each run it
reports the validation loss, the share of MLP units that never fired on a validation batch (dead
units, which matter for ReLU), and the time per step.

Uses F1's clean text and tokenizer by default, or characters from any text file with --text.

Output: out/swap_activation.png (validation loss curves) and out/swap_activation.json.
"""

import argparse
import json
import time

import torch

from common import ACTIVATIONS, GPT, OUT, GPTConfig, cosine_lr, count_params, load_corpus, pick_device

DEFAULT = ["none", "sigmoid", "tanh", "relu", "gelu", "silu", "swiglu"]


def dead_fraction(model, x):
    """Share of MLP hidden units whose activation output is exactly 0 for every token in x."""
    outs = []
    hooks = [b.mlp[1].register_forward_hook(lambda m, i, o: outs.append(o.detach()))
             for b in model.blocks if isinstance(b.mlp, torch.nn.Sequential)]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    if not outs:
        return None
    dead = [(o.flatten(0, 1) == 0).all(0).float().mean().item() for o in outs]
    return sum(dead) / len(dead)


def run(name, data, args, device, vocab):
    torch.manual_seed(args.seed)
    cfg = GPTConfig(vocab_size=vocab, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=4, n_embd=args.n_embd, activation=name)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    train, val = data
    gen = torch.Generator().manual_seed(args.seed)  # every run sees the same batches

    def batch(src, g=None):
        ix = torch.randint(len(src) - cfg.block_size - 1, (args.batch_size,), generator=g)
        x = torch.stack([src[i:i + cfg.block_size] for i in ix])
        y = torch.stack([src[i + 1:i + cfg.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    val_batches = [batch(val, torch.Generator().manual_seed(1000 + i)) for i in range(10)]

    @torch.no_grad()
    def val_loss():
        model.eval()
        loss = sum(model(x, y)[1].item() for x, y in val_batches) / len(val_batches)
        model.train()
        return loss

    curve = []
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.lr)
        _, loss = model(*batch(train, gen))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            curve.append((step, val_loss()))
    if device == "cuda":
        torch.cuda.synchronize()
    ms = 1000 * (time.time() - t0) / args.steps
    model.eval()
    dead = dead_fraction(model, val_batches[0][0])
    return {"activation": name, "params": count_params(model), "val_loss": curve[-1][1],
            "dead_units": dead, "ms_per_step": ms, "curve": curve}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activations", nargs="+", default=DEFAULT, choices=list(ACTIVATIONS))
    ap.add_argument("--text", help="train on characters from this text file instead of F1's data")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    ids, tok, source = load_corpus(args.text)
    split = int(0.9 * len(ids))
    data = (ids[:split], ids[split:])
    if len(data[1]) <= args.block_size + 1:
        raise SystemExit("Not enough text for a validation split. Lower --block-size or use more text.")
    print(f"{source}: {len(ids):,} tokens, vocab {tok.vocab_size:,}, device {device}, {args.steps} steps per run\n")
    print(f"{'activation':<11} {'params':>9} {'val loss':>9} {'dead units':>11} {'ms/step':>8}")

    results = []
    for name in args.activations:
        r = run(name, data, args, device, tok.vocab_size)
        results.append(r)
        dead = "n/a" if r["dead_units"] is None else f"{100 * r['dead_units']:.1f}%"
        print(f"{name:<11} {r['params']:>9,} {r['val_loss']:>9.3f} {dead:>11} {r['ms_per_step']:>8.1f}")

    OUT.mkdir(exist_ok=True)
    (OUT / "swap_activation.json").write_text(json.dumps({"source": source, "args": vars(args), "results": results}, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for r in results:
        steps, losses = zip(*r["curve"])
        ax.plot(steps, losses, lw=1.8, label=f"{r['activation']} ({r['val_loss']:.3f})")
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss")
    ax.set_title("Same tiny GPT, different MLP activation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "swap_activation.png", dpi=130)
    print(f"\nwrote {OUT / 'swap_activation.png'} and {OUT / 'swap_activation.json'}")


if __name__ == "__main__":
    main()
