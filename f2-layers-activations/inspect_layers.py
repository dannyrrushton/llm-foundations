"""Take the tiny GPT apart, layer by layer, and run it by hand.

1. The architecture: every layer, the shape it maps from and to, and how many parameters it holds.
2. A forward pass written out by hand, one operation at a time (embedding lookup, LayerNorm, the
   attention arithmetic, the MLP and its activation, residual adds, the output layer and softmax),
   printing the shape and spread of the data after each step, then checked against the model's own
   forward pass.
3. What the activation function actually sees: a histogram of the MLP's inputs to GELU in each block.
4. What comes out: the five most likely next tokens for a prompt.

Uses F1's trained model (f1-mini-pipeline/out/base.pt) and tokenizer when they exist; otherwise a
freshly initialized model of the same shape and random token ids, which still shows every shape and
check, just not meaningful predictions.

Output: out/mlp_preactivations.png, and the walkthrough printed to the terminal.
"""

import argparse
import math

import torch

from common import F1_CHECKPOINT, F1_TOKENIZER, GPT, OUT, F1Tokenizer, GPTConfig, count_params, load_f1_checkpoint

# The activation functions, by formula, for the hand-written pass.
BY_HAND = {
    "none": lambda x: x,
    "sigmoid": lambda x: 1 / (1 + torch.exp(-x)),
    "tanh": lambda x: (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x)),
    "relu": lambda x: torch.where(x > 0, x, torch.zeros_like(x)),
    "leaky_relu": lambda x: torch.where(x > 0, x, 0.01 * x),
    "gelu": lambda x: 0.5 * x * (1 + torch.erf(x / math.sqrt(2))),
    "gelu_tanh": lambda x: 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3))),
    "silu": lambda x: x / (1 + torch.exp(-x)),
}


def layer_table(model):
    cfg = model.cfg
    C, V, T = cfg.n_embd, cfg.vocab_size, cfg.block_size
    print(f"\n1. The architecture  ({cfg.n_layer} blocks, width {C}, {cfg.n_head} heads, vocab {V:,}, context {T}, activation {cfg.activation})\n")
    rows = [("tok", "Embedding", f"token id -> {C}", model.tok.weight.numel()),
            ("pos", "Embedding", f"position -> {C}", model.pos.weight.numel())]
    for i, b in enumerate(model.blocks):
        p = lambda m: sum(t.numel() for t in m.parameters())  # noqa: E731
        rows += [
            (f"blocks.{i}.ln1", "LayerNorm", f"{C} -> {C}", p(b.ln1)),
            (f"blocks.{i}.qkv", "Linear", f"{C} -> {3 * C} (q, k, v)", p(b.qkv)),
            (f"blocks.{i}.attn", "attention", f"{cfg.n_head} heads x {C // cfg.n_head}", 0),
            (f"blocks.{i}.proj", "Linear", f"{C} -> {C}", p(b.proj)),
            (f"blocks.{i}.ln2", "LayerNorm", f"{C} -> {C}", p(b.ln2)),
            (f"blocks.{i}.mlp", f"MLP ({cfg.activation})", f"{C} -> hidden -> {C}", p(b.mlp)),
        ]
    rows += [("ln_f", "LayerNorm", f"{C} -> {C}", model.ln_f.weight.numel() + model.ln_f.bias.numel()),
             ("head", "Linear (tied)", f"{C} -> {V:,} scores", 0),
             ("softmax", "softmax", f"{V:,} scores -> probabilities", 0)]
    for name, kind, shape, n in rows:
        print(f"  {name:<15} {kind:<15} {shape:<28} {n:>9,}")

    groups = {"embeddings": ("tok", "pos"), "attention": ("qkv", "proj"), "MLP": ("mlp",), "LayerNorm": ("ln",)}
    total = count_params(model)
    print(f"\n  {total:,} parameters in all. Where they live:")
    for g, keys in groups.items():
        n = sum(p.numel() for name, p in model.named_parameters() if any(k in name for k in keys) and "head" not in name)
        print(f"    {g:<11} {n:>9,}  {100 * n / total:5.1f}%")
    print("    head        shares the token-embedding matrix (weight tying), so it adds nothing")


def stat(name, t, note=""):
    print(f"  {name:<34} {str(tuple(t.shape)):<18} mean {t.mean().item():+.3f}  std {t.std().item():.3f}  {note}")


def layernorm(x, ln):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)  # population variance, as LayerNorm uses
    return (x - mu) / torch.sqrt(var + ln.eps) * ln.weight + ln.bias


def linear(x, lin):
    y = x @ lin.weight.T
    return y if lin.bias is None else y + lin.bias


def forward_by_hand(model, idx, record):
    cfg = model.cfg
    B, T = idx.shape
    C, H = cfg.n_embd, cfg.n_head
    d = C // H

    x = model.tok.weight[idx] + model.pos.weight[:T]          # look up rows; add position
    stat("embeddings (token + position)", x)
    for i, b in enumerate(model.blocks):
        print(f"  -- block {i} --")
        h = layernorm(x, b.ln1)
        stat("ln1", h, "<- every token row now mean 0, std 1, then scaled")
        q, k, v = linear(h, b.qkv).split(C, dim=2)
        q, k, v = (t.view(B, T, H, d).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(d)        # how well each query matches each key
        scores = scores.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))  # no peeking ahead
        weights = torch.softmax(scores, dim=-1)                 # each row: a distribution over earlier tokens
        y = (weights @ v).transpose(1, 2).reshape(B, T, C)
        a = linear(y, b.proj)
        stat("attention output (after proj)", a)
        x = x + a
        stat("residual stream after attention", x)
        h = layernorm(x, b.ln2)
        if cfg.activation == "swiglu":
            pre = linear(h, b.mlp.gate)
            post = BY_HAND["silu"](pre) * linear(h, b.mlp.up)
            m = linear(post, b.mlp.down)
        else:
            pre = linear(h, b.mlp[0])                           # expand to 4x width
            post = BY_HAND[cfg.activation](pre)                 # the non-linearity
            m = linear(post, b.mlp[2])                          # project back down
        record.append(pre.detach().flatten())
        neg = (pre < 0).float().mean().item()
        stat("MLP pre-activation", pre, f"{100 * neg:.0f}% negative")
        stat(f"MLP after {cfg.activation}", post, f"{100 * (post.abs() < 1e-2).float().mean().item():.0f}% near zero")
        x = x + m
        stat("residual stream after MLP", x)
    x = layernorm(x, model.ln_f)
    logits = x @ model.tok.weight.T                             # tied: the embedding matrix, used backwards
    stat("logits (one score per vocab entry)", logits)
    probs = torch.softmax(logits, dim=-1)
    stat("softmax probabilities", probs, f"rows sum to {probs[0, -1].sum().item():.6f}")
    return logits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(F1_CHECKPOINT), help="a checkpoint from F1's pretrain.py")
    ap.add_argument("--prompt", default="The universe is")
    args = ap.parse_args()
    torch.manual_seed(0)

    try:
        model, cfg = load_f1_checkpoint(args.checkpoint, "cpu")
        print(f"Loaded F1's trained model from {args.checkpoint}")
    except FileNotFoundError:
        cfg = GPTConfig()
        model = GPT(cfg)
        print("No F1 checkpoint found: using a freshly initialized model of the same shape.\n"
              "(Run F1 through pretrain.py for real weights and meaningful predictions.)")
    model = model.double().eval()  # float64, so the hand-written pass can match to ~1e-12

    tok = F1Tokenizer(F1_TOKENIZER) if F1_TOKENIZER.exists() else None
    if tok:
        idx = torch.tensor([tok.encode(args.prompt)])
        print(f"Prompt {args.prompt!r} -> {idx.shape[1]} tokens: {idx[0].tolist()}")
    else:
        idx = torch.randint(cfg.vocab_size, (1, 12))
        print(f"No F1 tokenizer: using 12 random token ids {idx[0].tolist()}")

    layer_table(model)

    print("\n2. One forward pass, by hand\n")
    record = []
    with torch.no_grad():
        mine = forward_by_hand(model, idx, record)
        theirs, _ = model(idx)
    print(f"\n  max |by hand - model(idx)| over all {mine.numel():,} logits: {(mine - theirs).abs().max().item():.2e}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(record), figsize=(3.6 * len(record), 3.4), sharey=True)
    axes = axes if len(record) > 1 else [axes]
    act = "silu" if cfg.activation == "swiglu" else cfg.activation
    xs = torch.linspace(-4, 4, 200, dtype=torch.float64)
    for i, (ax, pre) in enumerate(zip(axes, record)):
        ax.hist(pre.numpy(), bins=80, range=(-4, 4), density=True, alpha=0.6)
        ax2 = ax.twinx()
        ax2.axhline(0, color="0.6", lw=0.8, ls=":")
        ax2.plot(xs, BY_HAND[act](xs), color="black", lw=1.5)
        ax2.set_ylim(-1, 4)
        ax2.set_yticks([])
        ax.set_title(f"block {i}")
    fig.suptitle(f"What {act} sees: MLP pre-activations (histogram) and the function itself (line)")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "mlp_preactivations.png", dpi=130)
    print(f"\n3. wrote {OUT / 'mlp_preactivations.png'}")

    if tok:
        print(f"\n4. What comes next after {args.prompt!r}")
        p = torch.softmax(theirs[0, -1], -1)
        top = torch.topk(p, 5)
        for prob, i in zip(top.values, top.indices):
            print(f"  {prob.item():6.1%}  {tok.decode([i.item()])!r}")
        ent = -(p * p.clamp_min(1e-30).log()).sum().item()
        print(f"  entropy {ent:.2f} nats (a uniform guess over {cfg.vocab_size:,} would be {math.log(cfg.vocab_size):.2f})")


if __name__ == "__main__":
    main()
