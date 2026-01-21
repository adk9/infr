from __future__ import annotations

from typing import List


class ByteTokenizer:
    """Byte-level tokenizer mapping raw bytes to token ids 0-255.

    This keeps things deterministic and reversible for demo purposes.
    """

    def __init__(self, vocab_size: int = 256):
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 for byte tokenizer")
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: List[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="ignore")


def build_tokenizer(vocab_size: int) -> ByteTokenizer:
    effective_vocab = max(vocab_size, 256)
    return ByteTokenizer(vocab_size=effective_vocab)
