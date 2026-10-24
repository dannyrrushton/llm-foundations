"""Stage 7: pretrain a tiny GPT on the clean text: predict the next token, over and over.

The defaults (4 layers, 128-wide, about 1M parameters) train in minutes on a laptop CPU and faster
on a GPU. With only a few hundred thousand tokens the model will memorize as much as it learns;
watch the gap between train and validation loss.

Output: out/base.pt, plus a short sample of generated text.
"""

import argparse
import time

import torch

from common import CLEAN, END, OUT, GPTConfig, build_model, cosine_lr, count_params, pick_device, read_jsonl_dir, save_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    torch.manual_seed(args.seed)
    device = pick_device()
    tok = Tokenizer.from_file(str(OUT / "tokenizer.json"))
    end_id = tok.token_to_id(END)

    # One long stream of tokens, documents separated by <|end|>; the last 10% is held out.
    ids = []
    for doc in read_jsonl_dir(CLEAN):
        ids += tok.encode(doc["text"]).ids + [end_id]
    data = torch.tensor(ids, dtype=torch.long)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]
    if len(val) <= args.block_size + 1:
        raise SystemExit("Not enough text for a validation split. Lower --block-size or add pages.")

    cfg = GPTConfig(vocab_size=tok.get_vocab_size(), block_size=args.block_size,
                    n_layer=args.n_layer, n_head=4, n_embd=args.n_embd)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    print(f"{len(data):,} tokens ({len(train):,} train / {len(val):,} val) · "
          f"{count_params(model) / 1e6:.2f}M parameters · device {device}")

    def batch(src):
        ix = torch.randint(len(src) - cfg.block_size - 1, (args.batch_size,))
        x = torch.stack([src[i:i + cfg.block_size] for i in ix])
        y = torch.stack([src[i + 1:i + cfg.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def val_loss(n=20):
        model.eval()
        losses = [model(*batch(val))[1].item() for _ in range(n)]
        model.train()
        return sum(losses) / n

    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.lr)
        _, loss = model(*batch(train))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == args.steps - 1:
            print(f"step {step:>5}  train {loss.item():.3f}  val {val_loss():.3f}  {time.time() - t0:.0f}s")

    save_checkpoint(OUT / "base.pt", model, cfg)
    model.eval()
    prompt = torch.tensor([tok.encode("The universe").ids], device=device)
    print("\nSample:", tok.decode(model.generate(prompt, 80, stop_id=end_id)[0].tolist()))


if __name__ == "__main__":
    main()
