"""Shared pieces for F4: paths, the tiny GPT from F1 with a swappable activation, and data loading.

The model is the same ~1M-parameter GPT as F1's pretrain.py, with the same module names, so a
checkpoint trained there (f1-mini-pipeline/out/base.pt) loads here unchanged. The one addition is
that the MLP's activation function can be swapped, which is what swap_activation.py experiments with.
"""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE.parent))
from venv_check import warn_if_not_in_venv  # noqa: E402
from console import fail  # noqa: E402

warn_if_not_in_venv(HERE.name)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

OUT = HERE / "out"
F1 = HERE.parent / "f1-mini-pipeline"
F1_CHECKPOINT = F1 / "out" / "base.pt"
F1_TOKENIZER = F1 / "out" / "tokenizer.json"
F1_CLEAN = F1 / "data" / "clean"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Activation functions the MLP can use. "none" is the identity: no non-linearity at all.
# ---------------------------------------------------------------------------

ACTIVATIONS = {
    "none": nn.Identity,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "gelu": nn.GELU,                                   # exact (erf) form: what F1 and GPT-2's paper use
    "gelu_tanh": lambda: nn.GELU(approximate="tanh"),  # the fast approximation GPT-2's code uses
    "silu": nn.SiLU,                                   # also called Swish
    "swiglu": None,                                    # gated: built as its own MLP below
}


class SwiGLU(nn.Module):
    """The gated MLP used by LLaMA, Mistral and most recent open models (Shazeer, 2020).

    Two projections up: one passes through SiLU and acts as a gate, the other is multiplied by it.
    The hidden size is 8/3 of the width instead of 4x, so the parameter count matches a plain MLP.
    """

    def __init__(self, n_embd):
        super().__init__()
        hidden = 8 * ((8 * n_embd // 3 + 7) // 8)  # 8/3 x width, rounded up to a multiple of 8
        self.gate = nn.Linear(n_embd, hidden)
        self.up = nn.Linear(n_embd, hidden)
        self.down = nn.Linear(hidden, n_embd)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))


def build_mlp(n_embd, activation):
    if activation == "swiglu":
        return SwiGLU(n_embd)
    # Same layout as F1: mlp.0 (expand 4x), mlp.1 (activation), mlp.2 (project back).
    return nn.Sequential(nn.Linear(n_embd, 4 * n_embd), ACTIVATIONS[activation](), nn.Linear(4 * n_embd, n_embd))


# ---------------------------------------------------------------------------
# The tiny GPT: F1's model, written out as named layers.
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    activation: str = "gelu"


class Block(nn.Module):
    """One transformer block: LayerNorm, attention, residual add; LayerNorm, MLP, residual add."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = build_mlp(cfg.n_embd, cfg.activation)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        H = self.cfg.n_head
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.cfg.dropout if self.training else 0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))                  # residual add #1
        return x + self.drop(self.mlp(self.ln2(x)))      # residual add #2


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token id -> vector
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)   # position -> vector
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # vector -> one score per token
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss


def count_params(model):
    # Tied weights are one tensor, so count unique parameters.
    return sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())


def load_f1_checkpoint(path, device):
    """Load a checkpoint saved by F1's pretrain.py (or sft.py). Its config has no activation: it's GELU."""
    ckpt = torch.load(path, map_location=device)
    cfg = GPTConfig(**{**ckpt["config"], "activation": ckpt["config"].get("activation", "gelu")})
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device), cfg


# ---------------------------------------------------------------------------
# Data: F1's clean text and tokenizer if they exist, otherwise characters from any text file.
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Fallback tokenizer: one token per character. Enough to compare activations on equal terms."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    @property
    def vocab_size(self):
        return len(self.chars)


class F1Tokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(path))

    def encode(self, s):
        return self.tok.encode(s).ids

    def decode(self, ids):
        return self.tok.decode(ids)

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()


def read_f1_clean_text():
    import gzip

    docs = []
    for p in sorted(F1_CLEAN.rglob("*.jsonl*")):
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as f:
            docs += [json.loads(line)["text"] for line in f if line.strip()]
    return docs


def load_corpus(text_path=None):
    """Returns (token tensor, tokenizer, description of the source)."""
    if text_path:
        text = Path(text_path).read_text(encoding="utf-8")
        tok = CharTokenizer(text)
        return torch.tensor(tok.encode(text), dtype=torch.long), tok, f"{text_path} (characters)"
    docs = read_f1_clean_text() if F1_CLEAN.exists() else []
    if docs and F1_TOKENIZER.exists():
        tok = F1Tokenizer(F1_TOKENIZER)
        ids = []
        for d in docs:
            ids += tok.encode(d)
        return torch.tensor(ids, dtype=torch.long), tok, f"F1's clean text ({len(docs)} documents, BPE)"
    raise fail(
        "No training text. Either run F1 through train_tokenizer.py first (this uses its data/clean and\n"
        "tokenizer), or pass any plain-text file: --text path/to/book.txt"
    )


def cosine_lr(step, total, base_lr, warmup=50):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
