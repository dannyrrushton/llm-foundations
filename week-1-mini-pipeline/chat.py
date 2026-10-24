"""Stage 9: talk to the model. Type a question; an empty line or Ctrl-D quits.

Expect fluent-looking nonsense: a 1M-parameter model trained on forty pages knows very little.
The point is that every stage between raw web pages and this prompt has now been built once.
"""

import argparse

import torch

from common import ASSISTANT, END, OUT, USER, load_checkpoint, pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(OUT / "chat.pt"))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--prompt", help="ask one question and exit")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    device = pick_device()
    tok = Tokenizer.from_file(str(OUT / "tokenizer.json"))
    model, _, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    end_id = tok.token_to_id(END)

    def reply(question):
        ids = tok.encode(f"{USER}{question}{END}{ASSISTANT}").ids
        out = model.generate(torch.tensor([ids], device=device), args.max_tokens,
                             temperature=args.temperature, stop_id=end_id)[0].tolist()[len(ids):]
        return tok.decode([t for t in out if t != end_id]).strip()

    if args.prompt:
        print(reply(args.prompt))
        return
    while True:
        try:
            q = input("you> ").strip()
        except EOFError:
            break
        if not q:
            break
        print("model>", reply(q))


if __name__ == "__main__":
    main()
