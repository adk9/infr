from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict

from .config import ModelConfig


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def init_weights(config: ModelConfig, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = _rng(seed)
    weights: Dict[str, np.ndarray] = {}
    dtype = np.dtype(config.dtype)

    emb = rng.standard_normal((config.vocab_size, config.d_model), dtype=dtype) * np.asarray(0.02, dtype=dtype)
    pos = rng.standard_normal((config.max_seq_len, config.d_model), dtype=dtype) * np.asarray(0.02, dtype=dtype)
    weights["tok_embeddings"] = emb
    weights["pos_embeddings"] = pos

    for layer in range(config.n_layers):
        prefix = f"layers.{layer}."
        # Attention projections: Wqkv combines for simplicity; separate improves clarity.
        attn_scale = np.asarray(1.0 / np.sqrt(config.d_model), dtype=dtype)
        weights[prefix + "wq"] = rng.standard_normal((config.d_model, config.d_model), dtype=dtype) * attn_scale
        weights[prefix + "wk"] = rng.standard_normal((config.d_model, config.d_model), dtype=dtype) * attn_scale
        weights[prefix + "wv"] = rng.standard_normal((config.d_model, config.d_model), dtype=dtype) * attn_scale
        weights[prefix + "wo"] = rng.standard_normal((config.d_model, config.d_model), dtype=dtype) * attn_scale

        weights[prefix + "ln1_gamma"] = np.ones((config.d_model,), dtype=dtype)
        weights[prefix + "ln1_beta"] = np.zeros((config.d_model,), dtype=dtype)

        weights[prefix + "ln2_gamma"] = np.ones((config.d_model,), dtype=dtype)
        weights[prefix + "ln2_beta"] = np.zeros((config.d_model,), dtype=dtype)

        w1_scale = np.asarray(1.0 / np.sqrt(config.d_model), dtype=dtype)
        w2_scale = np.asarray(1.0 / np.sqrt(config.d_ff), dtype=dtype)
        weights[prefix + "w1"] = rng.standard_normal((config.d_model, config.d_ff), dtype=dtype) * w1_scale
        weights[prefix + "w2"] = rng.standard_normal((config.d_ff, config.d_model), dtype=dtype) * w2_scale
        weights[prefix + "b1"] = np.zeros((config.d_ff,), dtype=dtype)
        weights[prefix + "b2"] = np.zeros((config.d_model,), dtype=dtype)

    weights["ln_f_gamma"] = np.ones((config.d_model,), dtype=dtype)
    weights["ln_f_beta"] = np.zeros((config.d_model,), dtype=dtype)
    proj_scale = np.asarray(1.0 / np.sqrt(config.d_model), dtype=dtype)
    weights["output_projection"] = rng.standard_normal((config.d_model, config.vocab_size), dtype=dtype) * proj_scale
    return weights


def save_weights(weights: Dict[str, np.ndarray], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(Path(path), **weights)


def load_weights(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(Path(path))
    return {k: data[k] for k in data.files}
