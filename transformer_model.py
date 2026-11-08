"""
A standard decoder-only (GPT-style) character-level transformer, the baseline for
the spiking language model.

    tokens (B, T) -> token embedding + learned positions
                  -> n_layer x [ pre-LayerNorm causal multi-head self-attention,
                                 pre-LayerNorm GELU feed-forward ]  (residual connections)
                  -> LayerNorm -> output projection tied to the token embedding

It is trained with ordinary backpropagation. Its context is a window of
``max_len`` tokens; unlike the spiking network it has no state that survives
beyond that window.

``match_transformer`` picks the width and feed-forward size that make the
parameter count match a target (e.g. the spiking model's) as closely as the
architecture allows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class TransformerConfig:
    vocab_size: int
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int = 512
    max_len: int = 128
    dropout: float = 0.1
    tie_weights: bool = True        # output projection shares the token embedding (standard)


def param_count(c: TransformerConfig) -> int:
    """Exact parameter count, in closed form (checked against the real module)."""
    d, ff = c.d_model, c.d_ff
    per_layer = (2 * d                  # ln1
                 + 3 * d * d + 3 * d    # qkv
                 + d * d + d            # attention output projection
                 + 2 * d                # ln2
                 + d * ff + ff          # fc1
                 + ff * d + d)          # fc2
    total = c.vocab_size * d + c.max_len * d + c.n_layer * per_layer + 2 * d   # embeddings, blocks, final ln
    if not c.tie_weights:
        total += c.vocab_size * d
    return total


def match_transformer(target: int, vocab_size: int, max_len: int, n_layer: int = 4, n_head: int = 4,
                      dropout: float = 0.1) -> TransformerConfig:
    """Search the width ``d_model`` (a multiple of ``n_head``) and feed-forward size
    ``d_ff`` for the configuration closest to ``target`` parameters, preferring the
    conventional d_ff = 4 * d_model when several are about equally close."""
    best, best_score = None, float("inf")
    for d in range(2 * n_head, 4097, n_head):
        base = param_count(TransformerConfig(vocab_size, n_layer, n_head, d, 0, max_len, dropout))
        per_ff = n_layer * (2 * d + 1)                     # parameters added per unit of d_ff
        ff = int(round((target - base) / per_ff))
        ff = max(d // 2, min(8 * d, ff))
        cfg = TransformerConfig(vocab_size, n_layer, n_head, d, ff, max_len, dropout)
        rel = abs(param_count(cfg) - target) / target
        score = rel + 2e-3 * abs(math.log(ff / (4 * d)))
        if score < best_score:
            best, best_score = cfg, score
    return best


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head, self.p = n_head, dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0,
                                           is_causal=True)
        return self.resid_drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class Block(nn.Module):
    def __init__(self, c: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.attn = CausalSelfAttention(c.d_model, c.n_head, c.dropout)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.fc1 = nn.Linear(c.d_model, c.d_ff)
        self.fc2 = nn.Linear(c.d_ff, c.d_model)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))


class TransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)
        for name, p in self.named_parameters():          # GPT-2 scaling of the residual projections
            if name.endswith("attn.proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        assert self.num_parameters == param_count(cfg), (self.num_parameters, param_count(cfg))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())       # tied weights are counted once

    # The training loop treats both model types alike; the transformer carries no state.
    def reset_state(self) -> None:
        pass

    def detach_state(self) -> None:
        pass

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: (B, T) with T <= max_len. Returns next-token logits (B, T, V)."""
        T = tokens.shape[1]
        if T > self.cfg.max_len:
            raise ValueError(f"sequence length {T} exceeds the context window {self.cfg.max_len}")
        pos = torch.arange(T, device=tokens.device)
        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, prompt: Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, generator: Optional[torch.Generator] = None) -> Tensor:
        """Autoregressive sampling over a sliding window of the last ``max_len`` tokens."""
        was_training = self.training
        self.eval()
        ids = prompt.unsqueeze(0) if prompt.dim() == 1 else prompt
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.cfg.max_len:])[:, -1]
            if temperature <= 0.0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k < logits.shape[-1]:
                    kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            ids = torch.cat([ids, nxt], dim=1)
        self.train(was_training)
        return ids
