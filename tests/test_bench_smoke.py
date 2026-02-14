from toy_infer.config import ModelConfig
from toy_infer.weights import init_weights, save_weights
from toy_infer.runtime.engine import Engine
import numpy as np
import os

def test_bench_smoke(tmp_path):
    cfg = ModelConfig(vocab_size=128, d_model=32, n_heads=4, n_layers=1, d_ff=64, max_seq_len=32)
    weights = init_weights(cfg, seed=0)
    weights_path = tmp_path / "w.npz"
    save_weights(weights, weights_path)
    cfg_path = tmp_path / "c.json"
    cfg.dump(cfg_path)
    eng = Engine(str(cfg_path), str(weights_path), seed=0)
    text = eng.generate("hi", max_new_tokens=4)
    assert isinstance(text, str)
    assert len(text) > 0


def test_engine_generate_empty_prompt(tmp_path):
    cfg = ModelConfig(vocab_size=128, d_model=32, n_heads=4, n_layers=1, d_ff=64, max_seq_len=32)
    weights = init_weights(cfg, seed=0)
    weights_path = tmp_path / "w.npz"
    save_weights(weights, weights_path)
    cfg_path = tmp_path / "c.json"
    cfg.dump(cfg_path)

    eng = Engine(str(cfg_path), str(weights_path), seed=0)
    text = eng.generate("", max_new_tokens=4, greedy=True, temperature=0.0)

    assert isinstance(text, str)
    assert len(text) > 0
