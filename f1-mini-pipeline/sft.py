"""Stage 8: supervised fine-tuning, turning the base model into a (very small) chat model.

Each conversation is laid out in a chat template:

    <|user|>question<|end|><|assistant|>answer<|end|>

and the loss is computed on the assistant's tokens only, so the model learns to answer, not to ask.
Conversations come from HuggingFaceTB/smol-smoltalk (Apache-2.0), a set built for models under 1B
parameters. Only a few hundred are used, and they're truncated to the model's context length.

Output: out/chat.pt
"""

import argparse
import json
import time

import torch

from common import ASSISTANT, END, OUT, USER, cosine_lr, load_checkpoint, load_tokenizer, pick_device, require, save_checkpoint
from console import dim, heading, path  # common.py puts the repo root on sys.path


def load_conversations(n, path=None):
    if path:  # offline: a .jsonl file of {"messages": [{"role": ..., "content": ...}, ...]}
        with open(path, encoding="utf-8") as f:
            return [json.loads(line)["messages"] for line in f if line.strip()][:n]
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True)
    convs = []
    for row in ds:
        msgs = [m for m in row["messages"] if m["role"] in ("user", "assistant")]
        if len(msgs) >= 2:
            convs.append(msgs)
        if len(convs) >= n:
            break
    return convs


def encode(conv, tok, block_size):
    """Token ids plus targets, with -100 (ignored) everywhere except the assistant's replies."""
    ids, mask = [], []
    for m in conv:
        tag = USER if m["role"] == "user" else ASSISTANT
        head = tok.encode(tag).ids
        body = tok.encode(m["content"]).ids + [tok.token_to_id(END)]
        ids += head + body
        mask += [0] * len(head) + [1 if m["role"] == "assistant" else 0] * len(body)
    ids, mask = ids[: block_size + 1], mask[: block_size + 1]
    x = ids[:-1]
    y = [t if keep else -100 for t, keep in zip(ids[1:], mask[1:])]
    return x, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conversations", type=int, default=500)
    ap.add_argument("--data", help="local .jsonl of conversations instead of downloading smol-smoltalk")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = pick_device()
    tok = load_tokenizer()
    require(OUT / "base.pt", "pretrain.py")
    model, cfg, _ = load_checkpoint(OUT / "base.pt", device)

    pairs = [encode(c, tok, cfg.block_size) for c in load_conversations(args.conversations, args.data)]
    pairs = [p for p in pairs if any(t != -100 for t in p[1])]  # must contain some assistant text
    print(heading(f"{len(pairs)} conversations · device {device}"))

    def batch():
        pick = [pairs[i] for i in torch.randint(len(pairs), (args.batch_size,))]
        T = max(len(x) for x, _ in pick)
        X = torch.zeros(len(pick), T, dtype=torch.long)
        Y = torch.full((len(pick), T), -100, dtype=torch.long)
        for i, (x, y) in enumerate(pick):
            X[i, : len(x)], Y[i, : len(y)] = torch.tensor(x), torch.tensor(y)
        return X.to(device), Y.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.lr, warmup=20)
        _, loss = model(*batch())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"{dim('step')} {step:>5}  {dim('loss')} {loss.item():.3f}  {dim(f'{time.time() - t0:.0f}s')}")

    save_checkpoint(OUT / "chat.pt", model, cfg)
    print(f"Saved {path('out/chat.pt')}. Try it with: {heading('python chat.py')}")


if __name__ == "__main__":
    main()
