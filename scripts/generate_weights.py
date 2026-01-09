from __future__ import annotations

import typer
from pathlib import Path

from toy_infer.config import ModelConfig
from toy_infer.weights import init_weights, save_weights

app = typer.Typer(add_completion=False, help="Generate deterministic weights")


@app.command()
def main(
    config: str = typer.Option("configs/tiny.json", help="Config JSON"),
    out: str = typer.Option("artifacts/tiny.npz", help="Output weights"),
    seed: int = typer.Option(0, help="Random seed"),
):
    cfg = ModelConfig.load(config)
    weights = init_weights(cfg, seed=seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_weights(weights, out)
    typer.echo(f"Saved weights to {out}")


if __name__ == "__main__":
    app()
