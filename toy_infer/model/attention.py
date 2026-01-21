from __future__ import annotations

import numpy as np

from .layers import Linear


class KVCache:
    """Simple per-layer KV cache."""

    def __init__(self, batch: int, n_heads: int, max_seq_len: int, head_dim: int, dtype: str):
        self.keys = np.zeros((batch, n_heads, max_seq_len, head_dim), dtype=dtype)
        self.values = np.zeros((batch, n_heads, max_seq_len, head_dim), dtype=dtype)
        self.cur_pos = 0

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        # k,v: [batch, n_heads, 1, head_dim]
        idx = self.cur_pos
        self.keys[:, :, idx : idx + 1, :] = k
        self.values[:, :, idx : idx + 1, :] = v
        self.cur_pos += 1

    def slice(self, upto: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        end = self.cur_pos if upto is None else upto
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]


class MultiHeadSelfAttention:
    def __init__(self, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray, wo: np.ndarray, n_heads: int):
        self.wq = Linear(wq)
        self.wk = Linear(wk)
        self.wv = Linear(wv)
        self.wo = Linear(wo)
        self.n_heads = n_heads

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        # x: [batch, seq, d_model]
        batch, seq, d_model = x.shape
        head_dim = d_model // self.n_heads
        x = x.reshape(batch, seq, self.n_heads, head_dim)
        return np.transpose(x, (0, 2, 1, 3))  # [batch, heads, seq, head_dim]

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        # x: [batch, heads, seq, head_dim]
        batch, heads, seq, head_dim = x.shape
        x = np.transpose(x, (0, 2, 1, 3)).reshape(batch, seq, heads * head_dim)
        return x

    def _causal_mask(self, seq_len: int) -> np.ndarray:
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        return mask

    def __call__(
        self,
        x: np.ndarray,
        cache: KVCache | None,
        max_seq_len: int,
        decode: bool = False,
    ) -> tuple[np.ndarray, KVCache | None]:
        # x: [batch, seq, d_model]
        q = self._split_heads(self.wq(x))
        k = self._split_heads(self.wk(x))
        v = self._split_heads(self.wv(x))

        if cache is not None:
            if decode:
                # x is shape [batch, 1, d_model]
                cache.append(k, v)
                k_full, v_full = cache.slice()
                seq_len = cache.cur_pos
            else:
                # prefill
                cache.keys[:, :, : k.shape[2], :] = k
                cache.values[:, :, : v.shape[2], :] = v
                cache.cur_pos = k.shape[2]
                k_full, v_full = cache.slice()
                seq_len = k.shape[2]
        else:
            k_full, v_full = k, v
            seq_len = k.shape[2]

        attn_scores = np.matmul(q, np.transpose(k_full, (0, 1, 3, 2)))
        scale = np.asarray(1.0 / np.sqrt(q.shape[-1]), dtype=attn_scores.dtype)
        attn_scores *= scale

        if not decode:
            mask = self._causal_mask(seq_len)
            neg_inf = np.asarray(-1e9, dtype=attn_scores.dtype)
            attn_scores = np.where(mask[None, None, :, :], neg_inf, attn_scores)
        else:
            # Only need mask for last position in decode; earlier entries were cached.
            # Build mask for single query position (last token).
            mask = np.zeros((1, 1, 1, seq_len), dtype=bool)
            attn_scores[:, :, -1:, :] = np.where(mask, attn_scores[:, :, -1:, :], attn_scores[:, :, -1:, :])

        attn_weights = np.exp(attn_scores - attn_scores.max(axis=-1, keepdims=True))
        attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)

        context = np.matmul(attn_weights, v_full)  # [batch, heads, seq, head_dim]
        merged = self._merge_heads(context)
        out = self.wo(merged)
        return out, cache
