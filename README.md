# Toy Inference Baseline

A minimal, CPU-only transformer inference stack for optimization exercises. The code is intentionally simple and lightly optimized to expose performance opportunities.

## Quickstart
1) Create weights for the bundled tiny config:
```
python scripts/generate_weights.py --config configs/tiny.json --out artifacts/tiny.npz
```
2) Run a demo:
```
python scripts/run_demo.py --prompt "Hello" --max-new 32
```
3) Benchmark:
```
python -m toy_infer.bench.bench_cli --batch-size 4 --prompt-len 64 --gen-len 64 --requests 8 --profile
```
4) Run tests:
```
pytest
```

## Golden tensors (correctness baselines)
Golden tensors are saved outputs used to compare correctness as you implement optimizations. Generate them with the provided script after you have weights ready:

1) Generate weights (if you haven’t already):
```
python scripts/generate_weights.py --config configs/tiny.json --out artifacts/tiny.npz
```
2) Write golden outputs:
```
python scripts/verify_correctness.py write-golden --config configs/tiny.json --weights artifacts/tiny.npz --out artifacts/golden_outputs.npz
```

To compare current outputs against the golden tensors later:
```
python scripts/verify_correctness.py verify --config configs/tiny.json --weights artifacts/tiny.npz --golden artifacts/golden_outputs.npz
```

## Structure
- toy_infer/config.py — load/save model config
- toy_infer/tokenizer.py — byte-level tokenizer
- toy_infer/weights.py — deterministic init + load/save
- toy_infer/model/ — Linear, LayerNorm, attention, transformer blocks
- toy_infer/runtime/ — engine, sampling, scheduler abstraction
- toy_infer/bench/ — benchmarking harness + microbenchmarks
- configs/tiny.json — reference config
- scripts/generate_weights.py — produce weights
- scripts/run_demo.py — simple generation
- tests/ — correctness and sampling checks

## Generation API
```
from toy_infer.runtime.engine import Engine
eng = Engine("configs/tiny.json", "artifacts/tiny.npz")
text = eng.generate("Hello", max_new_tokens=16, temperature=0.8, top_k=20)
```

## Benchmark Metrics
- TTFT (time to first token)
- Decode latency (ms/token)
- Throughput (tokens/sec)
- Rough peak memory estimate
- Optional cProfile dump at artifacts/profile.pstats

## Baseline expectations
With the provided tiny config on a laptop CPU you should see roughly 20–200 tokens/sec. Performance is not the goal; improving it is.

## Notes
- CPU-only; dependencies: numpy, typer, rich (for pretty tables), pytest.
- Uses byte-level tokenizer (0–255) so prompts are reversible and deterministic.
- Deterministic weight init uses a fixed seed; override with `--seed` in generate_weights.py.
