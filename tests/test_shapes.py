import numpy as np

from toy_infer.config import ModelConfig
from toy_infer.weights import init_weights
from toy_infer.model.transformer import TransformerModel


def test_forward_shapes():
    cfg = ModelConfig(vocab_size=512, d_model=64, n_heads=4, n_layers=2, d_ff=128, max_seq_len=32)
    weights = init_weights(cfg, seed=1)
    model = TransformerModel(cfg, weights)
    tokens = np.zeros((2, 16), dtype=np.int64)
    logits = model.forward(tokens)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert logits.dtype == np.dtype(cfg.dtype)
