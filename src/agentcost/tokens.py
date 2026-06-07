"""Token estimation.

A real tokenizer (tiktoken, the Anthropic tokenizer, etc.) is the accurate way to count
tokens, but pulling one in is heavy and model-specific. agentcost ships with a tiny,
dependency-free heuristic (roughly four characters per token) that is good enough for
cost *attribution* -- where relative spend across features and users matters more than
exact billing -- and lets you plug in a precise tokenizer when you have one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Empirically ~4 characters per token for English text across common BPE tokenizers.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str, *, chars_per_token: int = CHARS_PER_TOKEN) -> int:
    """Estimate the number of tokens in ``text`` using a chars/N heuristic.

    Returns 0 for empty text and at least 1 token for any non-empty string.
    """
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    n = len(text)
    if n == 0:
        return 0
    return max(1, (n + chars_per_token - 1) // chars_per_token)


@runtime_checkable
class Tokenizer(Protocol):
    """Anything that can count tokens for a string.

    Implement this protocol to wrap a precise tokenizer (e.g. ``tiktoken``)::

        class TiktokenTokenizer:
            def __init__(self, enc):
                self._enc = enc

            def count(self, text: str) -> int:
                return len(self._enc.encode(text))
    """

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...


class HeuristicTokenizer:
    """A :class:`Tokenizer` backed by :func:`estimate_tokens`."""

    def __init__(self, chars_per_token: int = CHARS_PER_TOKEN) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """Return the estimated token count for ``text``."""
        return estimate_tokens(text, chars_per_token=self.chars_per_token)
