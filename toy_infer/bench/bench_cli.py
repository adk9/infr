from __future__ import annotations

import json
import cProfile
import pstats
import time
from pathlib import Path
from typing import List

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from ..runtime.engine import Engine
from ..runtime.scheduler import NaiveScheduler, Request
from ..config import ModelConfig
from .metrics import BenchResult, aggregate
from .microbench import run_micro

app = typer.Typer(add_completion=False, help="Benchmark CLI")
console = Console()


def peak_mem_estimate(config: ModelConfig, batch: int) -> int:
    bytes_per = np.dtype(config.dtype).itemsize
    kv = config.n_layers * 2 * batch * config.n_heads * config.max_seq_len * config.head_dim * bytes_per
    activations = batch * config.max_seq_len * config.d_model * bytes_per
    return kv + activations


def bench_once(engine: Engine, prompt_len: int, gen_len: int) -> BenchResult:
    # build random prompt
    rng = np.random.default_rng(0)
    prompt_tokens = rng.integers(0, engine.config.vocab_size, size=(1, prompt_len), dtype=np.int64)

    t0 = time.perf_counter()
    logits, cache = engine.prefill(prompt_tokens)
    ttft = (time.perf_counter() - t0) * 1000

    decode_times: List[float] = []
    last_logits = logits[:, -1, :]
    for _ in range(gen_len):
        t_step = time.perf_counter()
        next_token = np.argmax(last_logits, axis=-1)
        logits, cache = engine.decode_step(next_token[:, None], cache)
        last_logits = logits[:, -1, :]
        decode_times.append(time.perf_counter() - t_step)

    avg_decode_ms = (sum(decode_times) / len(decode_times)) * 1000 if decode_times else 0.0
    toks_sec = gen_len / sum(decode_times) if decode_times and sum(decode_times) > 0 else 0.0
    mem = peak_mem_estimate(engine.config, batch=1)
    return BenchResult(ttft, avg_decode_ms, toks_sec, mem)


def render_table(results: List[BenchResult]) -> None:
    table = Table(title="Benchmark Results")
    table.add_column("TTFT (ms)")
    table.add_column("Decode ms/token")
    table.add_column("Tokens/s")
    table.add_column("Peak mem (MB)")
    for r in results:
        table.add_row(
            f"{r.ttft_ms:.2f}",
            f"{r.decode_ms_per_token:.3f}",
            f"{r.tokens_per_sec:.1f}",
            f"{r.peak_mem_bytes/1e6:.2f}",
        )
    agg = aggregate(results)
    table.add_row("--", "--", "--", "--")
    table.add_row(
        f"{agg.ttft_ms:.2f}",
        f"{agg.decode_ms_per_token:.3f}",
        f"{agg.tokens_per_sec:.1f}",
        f"{agg.peak_mem_bytes/1e6:.2f}",
    )
    console.print(table)


@app.command()
def main(
    batch_size: int = typer.Option(1, help="Batch size"),
    prompt_len: int = typer.Option(64, help="Prompt length"),
    gen_len: int = typer.Option(64, help="Generation length"),
    requests: int = typer.Option(4, help="Number of requests"),
    config: str = typer.Option("configs/tiny.json", help="Config path"),
    weights: str = typer.Option("artifacts/tiny.npz", help="Weights path"),
    profile: bool = typer.Option(False, help="Write cProfile"),
    profile_path: str = typer.Option("artifacts/profile.pstats", help="Profile output"),
    micro: bool = typer.Option(False, help="Run microbenchmarks"),
):
    eng = Engine(config, weights)
    sched = NaiveScheduler()
    for _ in range(requests):
        sched.submit(Request(prompt="", max_new_tokens=gen_len, options={}))

    results: List[BenchResult] = []
    profiler = None
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    for _ in sched.run():
        res = bench_once(eng, prompt_len, gen_len)
        results.append(res)

    if profile and profiler:
        profiler.disable()
        Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(profile_path)
        console.print(f"Profile written to {profile_path}")

    render_table(results)
    out = [r.to_dict() for r in results]
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/bench.json").write_text(json.dumps(out, indent=2))
    console.print("Saved artifacts/bench.json")

    if micro:
        m = run_micro(ModelConfig.load(config))
        console.print({"microbench": m})


if __name__ == "__main__":
    app()
