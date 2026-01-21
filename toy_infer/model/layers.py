from __future__ import annotations

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    dtype = x.dtype
    half = np.asarray(0.5, dtype=dtype)
    one = np.asarray(1.0, dtype=dtype)
    coeff = np.asarray(0.044715, dtype=dtype)
    sqrt_two_over_pi = np.sqrt(np.asarray(2.0 / np.pi, dtype=dtype))
    return half * x * (one + np.tanh(sqrt_two_over_pi * (x + coeff * np.power(x, 3))))


class Linear:
    def __init__(self, weight: np.ndarray, bias: np.ndarray | None = None):
        self.weight = weight
        self.bias = bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class LayerNorm:
    def __init__(self, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
        self.gamma = gamma
        self.beta = beta
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        dtype = x.dtype
        mean = x.mean(axis=-1, keepdims=True, dtype=dtype)
        var = x.var(axis=-1, keepdims=True, dtype=dtype)
        eps = np.asarray(self.eps, dtype=dtype)
        norm = (x - mean) / np.sqrt(var + eps)
        return norm * self.gamma + self.beta
