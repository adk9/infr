import numpy as np

from toy_infer.config import ModelConfig
from toy_infer.weights import init_weights
from toy_infer.model.transformer import TransformerModel


def test_kv_cache_matches_full_forward():
    cfg = ModelConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=2, d_ff=128, max_seq_len=32)
    weights = init_weights(cfg, seed=2)
    model = TransformerModel(cfg, weights)

    tokens = np.arange(6, dtype=np.int64)[None, :]  # [1,6]
    prefill_len = 3
    prefill_tokens = tokens[:, :prefill_len]
    decode_tokens = tokens[:, prefill_len:]

    full_logits = model.forward(tokens)

    logits, caches = model.prefill(prefill_tokens)
    last_logits = logits[:, -1:, :]
    for t in decode_tokens.T:
        step = t[None, None]
        last_logits, caches = model.decode_step(step, caches)
    cache_logits = last_logits  # logits for last position

    np.testing.assert_allclose(cache_logits, full_logits[:, -1:, :], rtol=1e-4, atol=1e-4)
