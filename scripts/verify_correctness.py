from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy_infer.config import ModelConfig
from toy_infer.weights import load_weights
from toy_infer.model.transformer import TransformerModel
from toy_infer.runtime.engine import Engine

app = typer.Typer(add_completion=False, help="Generate or check golden outputs for correctness regression tests")

DEFAULT_PROMPTS = ["Hello", "byte-level", "", "The quick brown fox"]


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)


def _encode_prompts(prompts: List[str], cfg: ModelConfig) -> np.ndarray:
    from toy_infer.tokenizer import build_tokenizer

    tokenizer = build_tokenizer(cfg.vocab_size)
    encoded = [tokenizer.encode(p) for p in prompts]
    max_len = max(len(seq) for seq in encoded)
    arr = np.zeros((len(prompts), max_len), dtype=np.int64)
    for i, seq in enumerate(encoded):
        arr[i, : len(seq)] = seq
    return arr


def _run_forward(cfg: ModelConfig, weights_path: str, prompts: List[str]) -> Dict[str, np.ndarray]:
    _seed_everything(0)
    tokens = _encode_prompts(prompts, cfg)
    weights = load_weights(weights_path)
    model = TransformerModel(cfg, weights)
    logits = model.forward(tokens)
    return {"forward_logits": logits}


def _run_generation(cfg_path: str, weights_path: str, prompts: List[str], max_new: int) -> Dict[str, List[str]]:
    _seed_everything(0)
    eng = Engine(cfg_path, weights_path, seed=0)
    outputs = []
    for prompt in prompts:
        text = eng.generate(prompt, max_new_tokens=max_new, temperature=0.0, greedy=True)
        outputs.append(text)
    return {"generated_texts": outputs}


def _run_kv_equiv(cfg: ModelConfig, weights_path: str) -> Dict[str, np.ndarray]:
    _seed_everything(1)
    weights = load_weights(weights_path)
    model = TransformerModel(cfg, weights)
    tokens = np.arange(12, dtype=np.int64)[None, :]  # [1,12]
    prefill_len = 5
    prefill_tokens = tokens[:, :prefill_len]
    decode_tokens = tokens[:, prefill_len:]

    full_logits = model.forward(tokens)
    logits, cache = model.prefill(prefill_tokens)
    step_logits = logits[:, -1:, :]
    for t in decode_tokens.T:
        next_tok = t[None, None]
        step_logits, cache = model.decode_step(next_tok, cache)
    return {
        "kv_equiv_logits": step_logits,
        "full_tail_logits": full_logits[:, -1:, :],
    }


def _load_golden(path: Path) -> Dict[str, np.ndarray | List[str]]:
    with path.open("rb") as f:
        data = np.load(f, allow_pickle=True)
    out: Dict[str, np.ndarray | List[str]] = {}
    for key in data.files:
        val = data[key].item() if data[key].shape == () else data[key]
        out[key] = val
    return out


def _save_golden(path: Path, payload: Dict[str, np.ndarray | List[str]]) -> None:
    np.savez(path, **payload)


def _compare_arrays(name: str, ref: np.ndarray, cur: np.ndarray, atol: float, rtol: float) -> List[str]:
    try:
        np.testing.assert_allclose(cur, ref, rtol=rtol, atol=atol)
    except AssertionError as e:
        return [f"{name} mismatch: {e}"]
    return []


def _compare_texts(ref: List[str], cur: List[str]) -> List[str]:
    issues = []
    for i, (r, c) in enumerate(zip(ref, cur)):
        if r != c:
            issues.append(f"prompt {i} text mismatch: expected '{r}' got '{c}'")
    return issues


@app.command()
def write_golden(
    config: str = typer.Option("configs/tiny.json", help="Config JSON"),
    weights: str = typer.Option("artifacts/tiny.npz", help="Weights npz"),
    out: str = typer.Option("artifacts/golden_outputs.npz", help="Where to store golden outputs"),
    prompts: List[str] = typer.Option(DEFAULT_PROMPTS, help="Prompts to test"),
    max_new: int = typer.Option(8, help="Tokens to generate for each prompt"),
):
    cfg = ModelConfig.load(config)
    payload: Dict[str, np.ndarray | List[str]] = {}
    payload.update(_run_forward(cfg, weights, prompts))
    payload.update(_run_generation(config, weights, prompts, max_new))
    payload.update(_run_kv_equiv(cfg, weights))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    _save_golden(Path(out), payload)
    typer.echo(f"Wrote golden outputs to {out}")


@app.command()
def verify(
    config: str = typer.Option("configs/tiny.json", help="Config JSON"),
    weights: str = typer.Option("artifacts/tiny.npz", help="Weights npz"),
    golden: str = typer.Option("artifacts/golden_outputs.npz", help="Golden outputs produced by write_golden"),
    prompts: List[str] = typer.Option(DEFAULT_PROMPTS, help="Prompts to test"),
    max_new: int = typer.Option(8, help="Tokens to generate for each prompt"),
    atol: float = typer.Option(1e-4, help="Absolute tolerance for numeric checks"),
    rtol: float = typer.Option(1e-4, help="Relative tolerance for numeric checks"),
):
    cfg = ModelConfig.load(config)
    golden_path = Path(golden)
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden file not found: {golden}")

    golden_payload = _load_golden(golden_path)

    current: Dict[str, np.ndarray | List[str]] = {}
    current.update(_run_forward(cfg, weights, prompts))
    current.update(_run_generation(config, weights, prompts, max_new))
    current.update(_run_kv_equiv(cfg, weights))

    issues: List[str] = []
    issues += _compare_arrays("forward_logits", golden_payload["forward_logits"], current["forward_logits"], atol, rtol)
    issues += _compare_arrays("kv_equiv_logits", golden_payload["kv_equiv_logits"], current["kv_equiv_logits"], atol, rtol)
    issues += _compare_arrays("full_tail_logits", golden_payload["full_tail_logits"], current["full_tail_logits"], atol, rtol)
    issues += _compare_texts(golden_payload["generated_texts"].tolist(), current["generated_texts"])

    if issues:
        for line in issues:
            typer.secho(line, fg="red")
        raise typer.Exit(code=1)

    typer.secho("All checks passed", fg="green")


if __name__ == "__main__":
    app()
