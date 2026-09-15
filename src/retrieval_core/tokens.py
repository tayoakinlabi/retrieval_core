"""Counting tokens, before there is a tokenizer to count them with.

Chunking needs token counts and the real tokenizer arrives with the embedding
model (§1.1 item 2), which is a later increment. Rather than block on that, this
defines the interface chunking depends on and ships an estimate behind it.

**The estimate is not pretending to be exact, and the sizes do not need it to
be.** Strategy targets are approximate by design — "~400 tokens", "~500 tokens"
in §3.3 — and the consequence of being twenty tokens out is a chunk twenty
tokens longer, not a wrong answer. The one place exactness would matter is the
model's hard context ceiling, and §3.2 makes that a ceiling the targets sit far
below: with nomic v1.5's 8192 against a 400-token prose target, the estimate
would have to be off by a factor of twenty before it mattered.

What the estimate must not do is *under*-count, which is why it rounds up. An
under-count could push a chunk past the ceiling and have the model silently
truncate it — losing text that the index claims is there.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

__all__ = ["EstimatingCounter", "TokenCounter"]


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens the way the embedding model will, and knows its ceiling."""

    @property
    def max_context(self) -> int:
        """Most tokens the model accepts in one input."""
        ...

    def count(self, text: str) -> int: ...


# Word-ish runs, punctuation, and whitespace-separated symbols. A subword
# tokenizer splits long and unusual words further, which the multiplier below
# accounts for.
_WORDS = re.compile(r"\w+|[^\w\s]")

# Sub-word tokenizers emit roughly this many tokens per whitespace word on
# ordinary English prose. Rounded generously: see the module docstring on why
# over-counting is the safe direction.
_TOKENS_PER_WORD = 1.35


class EstimatingCounter:
    """A tokenizer-free estimate, replaced by the model's own when it lands.

    ``max_context`` defaults to nomic-embed-text-v1.5's 8192 (§6), the model the
    contract resolved on. A lite profile with a smaller window passes its own,
    which is exactly the case §3.2's ``min(model_max_context, strategy_target)``
    exists to handle.
    """

    def __init__(self, max_context: int = 8192) -> None:
        if max_context < 1:
            raise ValueError("max_context must be positive")
        self._max_context = max_context

    @property
    def max_context(self) -> int:
        return self._max_context

    def count(self, text: str) -> int:
        if not text:
            return 0
        pieces = _WORDS.findall(text)
        if not pieces:
            return 0
        # Long tokens fragment more than short ones, so weight by length rather
        # than counting pieces flat: one 30-character identifier is not one
        # token, and treating it as one is how an estimate under-counts.
        weighted = sum(max(1.0, len(piece) / 6.0) for piece in pieces)
        return max(1, int(weighted * _TOKENS_PER_WORD + 0.999))
