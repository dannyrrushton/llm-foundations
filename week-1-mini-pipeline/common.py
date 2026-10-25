"""Shared paths, the seed list reader, and a small GPT model used by every stage."""

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE.parent))
from venv_check import warn_if_not_in_venv  # noqa: E402

warn_if_not_in_venv(HERE.name)
DATA = HERE / "data"
RAW = DATA / "raw"              # fetch.py: one file per URL + manifest.jsonl
EXTRACTED = DATA / "extracted"  # extract_and_filter.py, stage 1 output (before dedup)
EXCLUDED = DATA / "excluded"    # one folder per filter: what it dropped
CLEAN = DATA / "clean"          # final, deduplicated text
MINHASH = DATA / "minhash"      # dedup working files
LOGS = DATA / "logs"
OUT = HERE / "out"              # tokenizer, checkpoints

# Special tokens for the chat format. They are added to the tokenizer in stage 6.
USER, ASSISTANT, END = "<|user|>", "<|assistant|>", "<|end|>"
SPECIAL_TOKENS = [END, USER, ASSISTANT]


def read_seed_list(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f, delimiter="\t") if row["url"].strip()]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_jsonl_dir(folder):
    """Every document in every .jsonl / .jsonl.gz file under a folder (datatrove's output format)."""
    import gzip

    docs = []
    for p in sorted(Path(folder).rglob("*.jsonl*")):
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as f:
            docs += [json.loads(line) for line in f if line.strip()]
    return docs


def load_tokenizer():
    from tokenizers import Tokenizer

    path = OUT / "tokenizer.json"
    if not path.exists():
        raise SystemExit("No out/tokenizer.json. Run train_tokenizer.py first.")
    return Tokenizer.from_file(str(path))


def require(path, previous_step):
    if not Path(path).exists():
        raise SystemExit(f"No {Path(path).relative_to(HERE)}. Run {previous_step} first.")


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# A small GPT, in the style of Karpathy's nanoGPT (github.com/karpathy/nanoGPT),
# cut down to the parts this toy run needs.
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


def build_model(cfg: GPTConfig):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.n_embd)
            self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
            self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
            self.ln2 = nn.LayerNorm(cfg.n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(), nn.Linear(4 * cfg.n_embd, cfg.n_embd)
            )
            self.drop = nn.Dropout(cfg.dropout)

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
            q, k, v = (t.view(B, T, cfg.n_head, C // cfg.n_head).transpose(1, 2) for t in (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=cfg.dropout if self.training else 0)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            x = x + self.drop(self.proj(y))
            return x + self.drop(self.mlp(self.ln2(x)))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList(Block() for _ in range(cfg.n_layer))
            self.ln_f = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
            self.head.weight = self.tok.weight  # weight tying, as in GPT-2
            self.apply(self._init)

        @staticmethod
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

        def forward(self, idx, targets=None):
            T = idx.shape[1]
            x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
            for b in self.blocks:
                x = b(x)
            logits = self.head(self.ln_f(x))
            loss = None
            if targets is not None:
                # targets == -100 are ignored: used in SFT to train on the assistant's turns only.
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss

        @torch.no_grad()
        def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, stop_id=None):
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -cfg.block_size:])
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
                idx = torch.cat([idx, nxt], dim=1)
                if stop_id is not None and nxt.item() == stop_id:
                    break
            return idx

    return GPT()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path, model, cfg, **extra):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, **extra}, path)


def load_checkpoint(path, device):
    import torch

    ckpt = torch.load(path, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device), cfg, ckpt


def cosine_lr(step, total, base_lr, warmup=50):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
