import numpy as np

from toy_infer.runtime.sampling import top_k_filter, top_p_filter, greedy_from_logits, sample_from_logits


def test_top_k_filters_only_k():
    logits = np.array([[1.0, 2.0, 3.0, -1.0]])
    filtered = top_k_filter(logits, k=2)
    # only top2 should survive
    assert filtered[0, 2] > -1e8
    assert filtered[0, 1] > -1e8
    assert filtered[0, 0] < -1e8 or filtered[0, 3] < -1e8


def test_top_p_keeps_mass():
    logits = np.array([[3.0, 2.0, 1.0, 0.0]])
    filtered = top_p_filter(logits, p=0.6)
    # highest logit should remain
    assert filtered[0, 0] > -1e8


def test_greedy_and_sample_shapes():
    logits = np.array([[0.1, 0.2, 0.6]], dtype=np.float32)
    g = greedy_from_logits(logits)
    assert g.shape == (1,)
    s = sample_from_logits(logits, temperature=1.0)
    assert s.shape == (1,)
