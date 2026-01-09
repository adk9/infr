from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BenchResult:
    ttft_ms: float
    decode_ms_per_token: float
    tokens_per_sec: float
    peak_mem_bytes: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "ttft_ms": self.ttft_ms,
            "decode_ms_per_token": self.decode_ms_per_token,
            "tokens_per_sec": self.tokens_per_sec,
            "peak_mem_bytes": self.peak_mem_bytes,
        }


def aggregate(results: List[BenchResult]) -> BenchResult:
    n = len(results)
    return BenchResult(
        ttft_ms=sum(r.ttft_ms for r in results) / n,
        decode_ms_per_token=sum(r.decode_ms_per_token for r in results) / n,
        tokens_per_sec=sum(r.tokens_per_sec for r in results) / n,
        peak_mem_bytes=max(r.peak_mem_bytes for r in results),
    )
