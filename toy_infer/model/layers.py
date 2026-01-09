from __future__ import annotations

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


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
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        norm = (x - mean) / np.sqrt(var + self.eps)
        return norm * self.gamma + self.beta
