"""Stage 6: train a small byte-level BPE tokenizer on the clean text.

Byte-level BPE is the GPT-2 family's scheme: start from the 256 byte values and repeatedly merge the
most frequent adjacent pair until the vocabulary is full. The chat special tokens are added now so
they each stay a single token.

Output: out/tokenizer.json
"""

import argparse

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from common import CLEAN, OUT, SPECIAL_TOKENS, read_jsonl_dir
from console import dim, fail, heading  # common.py puts the repo root on sys.path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab-size", type=int, default=4096)
    args = ap.parse_args()

    texts = [d["text"] for d in read_jsonl_dir(CLEAN)]
    if not texts:
        raise fail("No documents in data/clean/. Run extract_and_filter.py first.")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)

    OUT.mkdir(parents=True, exist_ok=True)
    tok.save(str(OUT / "tokenizer.json"))

    n_chars = sum(len(t) for t in texts)
    n_tokens = sum(len(tok.encode(t).ids) for t in texts)
    sample = "Quantum mechanics describes how very small things behave."
    print(heading(f"Vocabulary {tok.get_vocab_size()} · {len(texts)} documents · {n_chars:,} characters -> "
          f"{n_tokens:,} tokens ({n_chars / n_tokens:.1f} characters per token)"))
    print(f"{dim('Example:')} {tok.encode(sample).tokens}")


if __name__ == "__main__":
    main()
