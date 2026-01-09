from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return logits
    kth = np.partition(logits, -k, axis=-1)[..., -k][..., None]
    filtered = np.where(logits < kth, -1e9, logits)
    return filtered


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    if p <= 0 or p >= 1:
        return logits
    sorted_idx = np.argsort(logits)[..., ::-1]
    sorted_logits = np.take_along_axis(logits, sorted_idx, axis=-1)
    probs = softmax(sorted_logits)
    cum = np.cumsum(probs, axis=-1)
    mask = cum > p
    # ensure at least one token
    mask[..., 0] = False
    sorted_logits = np.where(mask, -1e9, sorted_logits)
    return np.take_along_axis(sorted_logits, np.argsort(sorted_idx, axis=-1), axis=-1)


def sample_from_logits(logits: np.ndarray, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None) -> np.ndarray:
    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = softmax(logits)
    batch = probs.shape[0]
    vocab = probs.shape[1]
    draws = np.empty(batch, dtype=np.int64)
    for i in range(batch):
        draws[i] = np.random.choice(vocab, p=probs[i])
    return draws


def greedy_from_logits(logits: np.ndarray) -> np.ndarray:
    return np.argmax(logits, axis=-1)
