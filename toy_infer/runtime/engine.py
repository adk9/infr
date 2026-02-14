from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import typer

from ..config import ModelConfig
from ..tokenizer import build_tokenizer
from ..weights import load_weights
from ..model.transformer import TransformerModel
from .sampling import greedy_from_logits, sample_from_logits

app = typer.Typer(add_completion=False, help="Toy inference engine")


class Engine:
    def __init__(self, config_path: str, weights_path: str, seed: int = 0):
        self.config = ModelConfig.load(config_path)
        self.tokenizer = build_tokenizer(self.config.vocab_size)
        self.weights = load_weights(weights_path)
        np.random.seed(seed)
        self.model = TransformerModel(self.config, self.weights)

    def prefill(self, prompt_tokens: np.ndarray):
        logits, cache = self.model.prefill(prompt_tokens)
        return logits, cache

    def decode_step(self, token: np.ndarray, cache):
        logits, cache = self.model.decode_step(token, cache)
        return logits, cache

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        greedy: bool = False,
    ) -> str:
        encoded_prompt = self.tokenizer.encode(prompt)
        if encoded_prompt:
            tokens = np.array([encoded_prompt], dtype=np.int64)
            logits, cache = self.prefill(tokens)
            out_tokens: List[int] = encoded_prompt.copy()
        else:
            bootstrap = np.zeros((1, 1), dtype=np.int64)
            logits, cache = self.prefill(bootstrap)
            out_tokens = []

        for _ in range(max_new_tokens):
            last_logits = logits[:, -1, :]  # [batch, vocab]
            if greedy or temperature == 0:
                next_token = greedy_from_logits(last_logits)
            else:
                next_token = sample_from_logits(
                    last_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            out_tokens.append(int(next_token[0]))
            next_arr = np.array([[next_token[0]]], dtype=np.int64)
            logits, cache = self.decode_step(next_arr, cache)
        return self.tokenizer.decode(out_tokens)


@app.command()
def main(
    prompt: str = typer.Option("Hello", help="Prompt text"),
    max_new: int = typer.Option(32, help="Number of tokens to generate"),
    config: str = typer.Option("configs/tiny.json", help="Path to config"),
    weights: str = typer.Option("artifacts/tiny.npz", help="Path to weights"),
    temperature: float = typer.Option(1.0, help="Sampling temperature"),
    top_k: int = typer.Option(0, help="Top-k (0 to disable)"),
    top_p: float = typer.Option(0.0, help="Top-p (0 to disable)"),
    greedy: bool = typer.Option(False, help="Use greedy decoding"),
):
    eng = Engine(config, weights)
    start = time.perf_counter()
    text = eng.generate(
        prompt,
        max_new_tokens=max_new,
        temperature=temperature,
        top_k=top_k or None,
        top_p=top_p or None,
        greedy=greedy,
    )
    elapsed = time.perf_counter() - start
    typer.secho(text, fg="green")
    typer.echo(f"Generated in {elapsed:.3f}s")


if __name__ == "__main__":
    app()
