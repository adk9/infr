from __future__ import annotations

import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy_infer.runtime.engine import Engine

app = typer.Typer(add_completion=False, help="Run a quick demo")


@app.command()
def main(
    prompt: str = typer.Option("Hello world", help="Prompt"),
    max_new: int = typer.Option(32, help="Tokens to generate"),
    config: str = typer.Option("configs/tiny.json"),
    weights: str = typer.Option("artifacts/tiny.npz"),
):
    eng = Engine(config, weights)
    text = eng.generate(prompt, max_new_tokens=max_new)
    typer.echo(text)


if __name__ == "__main__":
    app()
