"""The input boundary. **The library never opens a source file** (contract §1.2).

Products parse — PDFs, audio, screen recordings, spreadsheets — and hand in
spans. That line is the whole design, and §1's rationale says why: parsing is
the fragile part. A Whisper break, a PDF library break, a screen-capture break
should cost one product, not all three. Moving *chunking* in (L6) deliberately
did not move parsing in with it.

A span is a piece of text that already knows where it came from. What the
library adds is the decision about where to cut it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from retrieval_core.provenance import Locator

__all__ = ["Span", "SpanKind"]


class SpanKind(StrEnum):
    """What this text *is*, which decides how it may be cut.

    Not a formatting hint. ``transcript_turn`` means one person speaking once
    and carries the rule that it is never split mid-turn; ``step`` means one
    step of a procedure and carries the rule that steps never merge. The kind is
    how a parser tells the library which rules apply.
    """

    PROSE = "prose"
    TRANSCRIPT_TURN = "transcript_turn"
    STEP = "step"
    TABLE_ROW = "table_row"
    HEADING = "heading"


@dataclass(frozen=True, slots=True)
class Span:
    """One parsed piece of a source, with the locator the parser worked out."""

    text: str
    locator: Locator
    kind: SpanKind = SpanKind.PROSE
    # Speaker, channel, heading level — whatever the parser knows and the
    # strategies or the product may want. Not interpreted here.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("span text must be a string")
