from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List


@dataclass
class Request:
    prompt: str
    max_new_tokens: int
    options: dict[str, Any]


class Scheduler:
    """Abstraction to allow alternative batching strategies.

    Baseline simply processes requests sequentially.
    """

    def __init__(self):
        self.queue: List[Request] = []

    def submit(self, req: Request) -> None:
        self.queue.append(req)

    def run(self) -> Iterable[Request]:
        while self.queue:
            yield self.queue.pop(0)


class NaiveScheduler(Scheduler):
    pass
