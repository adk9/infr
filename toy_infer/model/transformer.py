from __future__ import annotations

import numpy as np

from .attention import KVCache, MultiHeadSelfAttention
from .layers import LayerNorm, Linear, gelu
from ..config import ModelConfig


class TransformerBlock:
    def __init__(self, config: ModelConfig, weights: dict, layer_idx: int):
        p = f"layers.{layer_idx}."
        self.ln1 = LayerNorm(weights[p + "ln1_gamma"], weights[p + "ln1_beta"])
        self.ln2 = LayerNorm(weights[p + "ln2_gamma"], weights[p + "ln2_beta"])
        self.attn = MultiHeadSelfAttention(
            wq=weights[p + "wq"],
            wk=weights[p + "wk"],
            wv=weights[p + "wv"],
            wo=weights[p + "wo"],
            n_heads=config.n_heads,
        )
        self.ff1 = Linear(weights[p + "w1"], weights[p + "b1"])
        self.ff2 = Linear(weights[p + "w2"], weights[p + "b2"])

    def __call__(self, x: np.ndarray, cache: KVCache | None, config: ModelConfig, decode: bool) -> tuple[np.ndarray, KVCache | None]:
        y, cache = self.attn(self.ln1(x), cache, max_seq_len=config.max_seq_len, decode=decode)
        x = x + y
        y = self.ff2(gelu(self.ff1(self.ln2(x))))
        x = x + y
        return x, cache


class TransformerModel:
    def __init__(self, config: ModelConfig, weights: dict):
        self.config = config
        self.weights = weights
        self.blocks = [TransformerBlock(config, weights, i) for i in range(config.n_layers)]
        self.tok_embeddings = weights["tok_embeddings"]
        self.pos_embeddings = weights["pos_embeddings"]
        self.ln_f = LayerNorm(weights["ln_f_gamma"], weights["ln_f_beta"])
        self.output_proj = Linear(weights["output_projection"])

    def _embed(self, tokens: np.ndarray) -> np.ndarray:
        batch, seq_len = tokens.shape
        tok = self.tok_embeddings[tokens]
        pos = self.pos_embeddings[np.arange(seq_len)]
        pos = np.broadcast_to(pos, tok.shape)
        return tok + pos

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        x = self._embed(tokens)
        cache = None
        for block in self.blocks:
            x, cache = block(x, cache, self.config, decode=False)
        x = self.ln_f(x)
        logits = self.output_proj(x)
        return logits

    def prefill(self, tokens: np.ndarray) -> tuple[np.ndarray, list[KVCache]]:
        x = self._embed(tokens)
        caches = [KVCache(tokens.shape[0], self.config.n_heads, self.config.max_seq_len, self.config.head_dim, self.config.dtype) for _ in range(self.config.n_layers)]
        for i, block in enumerate(self.blocks):
            x, caches[i] = block(x, caches[i], self.config, decode=False)
        x = self.ln_f(x)
        logits = self.output_proj(x)
        return logits, caches

    def decode_step(self, token: np.ndarray, caches: list[KVCache]) -> tuple[np.ndarray, list[KVCache]]:
        # token: [batch, 1]
        x = self._embed(token)
        for i, block in enumerate(self.blocks):
            x, caches[i] = block(x, caches[i], self.config, decode=True)
        x = self.ln_f(x)
        logits = self.output_proj(x)
        return logits, caches
