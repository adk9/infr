from __future__ import annotations

import time
from typing import Dict

import numpy as np

from ..config import ModelConfig
from ..weights import init_weights
from ..model.layers import LayerNorm
from ..model.attention import KVCache


def bench_attention(config: ModelConfig, batch: int = 1, seq: int = 64) -> float:
    weights = init_weights(config, seed=42)
    kvc = KVCache(batch, config.n_heads, config.max_seq_len, config.head_dim, config.dtype)
    q = np.random.randn(batch, seq, config.d_model).astype(config.dtype)
    k = np.random.randn(batch, seq, config.d_model).astype(config.dtype)
    v = np.random.randn(batch, seq, config.d_model).astype(config.dtype)
    start = time.perf_counter()
    scores = np.matmul(q, np.transpose(k, (0, 2, 1)))  # crude hotspot
    _ = scores @ v
    kvc.keys[:, :, :seq, :] = k.reshape(batch, config.n_heads, seq, config.head_dim)
    kvc.values[:, :, :seq, :] = v.reshape(batch, config.n_heads, seq, config.head_dim)
    return time.perf_counter() - start


def bench_layernorm(config: ModelConfig, batch: int = 1, seq: int = 64) -> float:
    ln = LayerNorm(np.ones(config.d_model), np.zeros(config.d_model))
    x = np.random.randn(batch, seq, config.d_model).astype(config.dtype)
    start = time.perf_counter()
    _ = ln(x)
    return time.perf_counter() - start


def bench_matmul(shapes=((64, 256, 256), (256, 256))) -> float:
    a_shape, b_shape = shapes
    a = np.random.randn(*a_shape).astype("float32")
    b = np.random.randn(*b_shape).astype("float32")
    start = time.perf_counter()
    _ = a @ b
    return time.perf_counter() - start


def run_micro(config: ModelConfig) -> Dict[str, float]:
    return {
        "attention": bench_attention(config),
        "layernorm": bench_layernorm(config),
        "matmul": bench_matmul(),
    }
