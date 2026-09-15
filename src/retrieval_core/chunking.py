"""Cutting spans into chunks — contract §3, and the reason the boundary is medium.

L6 moved chunking *into* the library and deliberately left parsing out. The two
look similar and are not: parsing is where formats break, chunking is where
retrieval quality is decided. Three products making three slightly different
guesses about chunk size would give three different retrieval qualities for no
reason anybody could name.

Two rules run through everything here.

**Model context is a ceiling, not a target** (§3.2). ``chunk_size = min(model
context, strategy target)``. Filling nomic's 8192-token window with one chunk
would dilute the embedding until nothing distinctive survives, destroy the
citation granularity Knowhow is differentiated on, and match many queries weakly
instead of few strongly. The large window is valuable because it means nothing is
ever *forced* to fragment — a long speaker turn or a verbose step can stay whole.

**A chunk's locator points at the chunk** (§2.1 rule 4). If a span is cut, its
locator is cut with it; if spans are merged, their locators are merged. A
citation that points at the whole span a chunk came from is a citation that
sends the reader to the wrong paragraph.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from retrieval_core.provenance import (
    CellLocator,
    FileLocator,
    Locator,
    ProvenanceRecord,
    SourceType,
    TimeLocator,
)
from retrieval_core.spans import Span, SpanKind
from retrieval_core.tokens import EstimatingCounter, TokenCounter

__all__ = [
    "STRATEGIES",
    "Chunk",
    "ChunkingConfig",
    "Source",
    "chunk_spans",
]

STRATEGIES = ("prose", "transcript", "steps", "tabular")


@dataclass(frozen=True, slots=True)
class Source:
    """What the product knows about the artifact these spans came from.

    Separate from :class:`Span` because it is the same for every span of one
    file, and separate from the library because only the product can compute it
    — the library never opens the source (§1.2).
    """

    source_id: str
    source_type: SourceType
    source_uri: str
    source_title: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """One unit of retrieval, carrying everything needed to cite it."""

    chunk_id: str
    text: str
    provenance: ProvenanceRecord
    token_count: int

    def cite(self) -> str:
        return self.provenance.cite()


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Targets from §3.3. All configurable, as the contract requires.

    ``prose_overlap`` is a fraction; the others are counts. Overlap is applied
    by repeating whole spans rather than slicing text, which is what keeps
    "avoid splitting mid-paragraph" and truthful locators compatible — an
    overlap made of half a paragraph would have to claim a locator it does not
    own.
    """

    prose_target: int = 400
    prose_overlap: float = 0.15
    transcript_target: int = 500
    transcript_overlap_turns: int = 1
    tabular_rows: int = 20
    # Set by the caller when small-to-big expansion is wanted (§3.4): chunks get
    # a parent_id so retrieval can match precisely and then hand generation the
    # wider section.
    parent_id: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.prose_overlap < 1.0:
            raise ValueError("prose_overlap must be a fraction below 1")
        for name in ("prose_target", "transcript_target", "tabular_rows"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


# Sentence ends, for the hard split of last resort. Deliberately conservative:
# a split in the wrong place costs a little retrieval quality, and the
# alternative — cutting mid-word — costs more.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\s]+|\n{2,}")


def chunk_spans(
    spans: list[Span],
    source: Source,
    *,
    strategy: str = "prose",
    config: ChunkingConfig | None = None,
    counter: TokenCounter | None = None,
    ingested_at: datetime | None = None,
) -> list[Chunk]:
    """Cut spans into chunks under one strategy.

    Raises ``ValueError`` for an unknown strategy rather than falling back to
    prose: a caller that asked for ``transcript`` and silently got prose would
    get turns split mid-sentence and never find out why retrieval got worse.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {', '.join(STRATEGIES)}")
    config = config or ChunkingConfig()
    counter = counter or EstimatingCounter()
    ingested_at = ingested_at or datetime.now(UTC)

    if strategy == "prose":
        groups = _group_prose(spans, config, counter)
    elif strategy == "transcript":
        groups = _group_transcript(spans, config, counter)
    elif strategy == "steps":
        groups = _group_steps(spans, config, counter)
    else:
        groups = _group_tabular(spans, config, counter)

    return [
        _build_chunk(group, source, config, counter, ingested_at, index)
        for index, group in enumerate(groups)
        if group
    ]


def _limit(target: int, counter: TokenCounter) -> int:
    """§3.2, applied: the ceiling never raises a target, only lowers it."""
    return min(counter.max_context, target)


def _group_prose(
    spans: list[Span], config: ChunkingConfig, counter: TokenCounter
) -> list[list[Span]]:
    """Pack spans up to the target, overlapping by whole spans."""
    limit = _limit(config.prose_target, counter)
    overlap_budget = int(limit * config.prose_overlap)

    groups: list[list[Span]] = []
    current: list[Span] = []
    used = 0

    for span in spans:
        for piece in _split_if_oversized(span, limit, counter):
            size = counter.count(piece.text)
            if current and used + size > limit:
                groups.append(current)
                current = _tail_within(current, overlap_budget, counter)
                used = sum(counter.count(s.text) for s in current)
            current.append(piece)
            used += size

    if current:
        groups.append(current)
    return groups


def _tail_within(group: list[Span], budget: int, counter: TokenCounter) -> list[Span]:
    """The last whole spans of a group that fit in the overlap budget."""
    if budget <= 0:
        return []
    tail: list[Span] = []
    total = 0
    for span in reversed(group):
        size = counter.count(span.text)
        if total + size > budget:
            break
        tail.insert(0, span)
        total += size
    return tail


def _group_transcript(
    spans: list[Span], config: ChunkingConfig, counter: TokenCounter
) -> list[list[Span]]:
    """Pack turns up to the target. **Never split mid-turn** (§3.3).

    A turn that alone exceeds the ceiling is hard-split, because the alternative
    is handing the model more than it accepts and having it truncate silently.
    Everything short of that stays whole: half of "no, we agreed the opposite"
    retrieved without the other half is worse than useless.
    """
    limit = _limit(config.transcript_target, counter)
    groups: list[list[Span]] = []
    current: list[Span] = []
    used = 0

    for span in spans:
        pieces = _split_if_oversized(span, limit, counter)
        for piece in pieces:
            size = counter.count(piece.text)
            if current and used + size > limit:
                groups.append(current)
                # Overlap is whole turns, so a chunk always starts at the start
                # of something somebody said.
                current = (
                    current[-config.transcript_overlap_turns :]
                    if config.transcript_overlap_turns
                    else []
                )
                used = sum(counter.count(s.text) for s in current)
            current.append(piece)
            used += size

    if current:
        groups.append(current)
    return groups


def _group_steps(
    spans: list[Span], config: ChunkingConfig, counter: TokenCounter
) -> list[list[Span]]:
    """One step, one chunk. **Never merged across steps** (§3.3).

    Merging two steps into one chunk would make a procedure's citation point at
    "steps 4 and 5", which is precisely the granularity a person following
    instructions cannot use.
    """
    limit = counter.max_context
    groups: list[list[Span]] = []
    for span in spans:
        for piece in _split_if_oversized(span, limit, counter):
            groups.append([piece])
    return groups


def _group_tabular(
    spans: list[Span], config: ChunkingConfig, counter: TokenCounter
) -> list[list[Span]]:
    """Header plus N rows, header repeated, **never splitting a row** (§3.3).

    The header is repeated into every chunk because a row of figures without its
    column names is not retrievable by anything a person would type — "revenue
    Q3" has to match a chunk that contains the word revenue.
    """
    limit = counter.max_context
    headers = [s for s in spans if s.kind is SpanKind.HEADING]
    rows = [s for s in spans if s.kind is not SpanKind.HEADING]
    header_cost = sum(counter.count(s.text) for s in headers)

    groups: list[list[Span]] = []
    current: list[Span] = []
    used = header_cost

    for row in rows:
        # A row is never split, so an oversized one becomes its own chunk and
        # stays whole even past the target.
        size = counter.count(row.text)
        too_many = len(current) >= config.tabular_rows
        too_big = current and used + size > limit
        if too_many or too_big:
            groups.append(headers + current)
            current = []
            used = header_cost
        current.append(row)
        used += size

    if current:
        groups.append(headers + current)
    return groups


def _split_if_oversized(span: Span, limit: int, counter: TokenCounter) -> list[Span]:
    """Hard-split a span that cannot fit, at the nearest sentence boundary (§3.3).

    Only ever reached by a *single* span larger than the limit on its own. Every
    strategy above prefers to start a new chunk instead; this is what happens
    when there is no smaller unit to fall back on.
    """
    if counter.count(span.text) <= limit:
        return [span]

    sentences = _sentences(span.text)
    pieces: list[Span] = []
    buffer: list[str] = []
    buffer_start = 0
    used = 0

    for text, offset in sentences:
        size = counter.count(text)
        if buffer and used + size > limit:
            joined = "".join(buffer)
            pieces.append(_subdivide(span, buffer_start, buffer_start + len(joined), joined))
            buffer, used = [], 0
            buffer_start = offset
        if not buffer:
            buffer_start = offset
        buffer.append(text)
        used += size

    if buffer:
        joined = "".join(buffer)
        pieces.append(_subdivide(span, buffer_start, buffer_start + len(joined), joined))
    return pieces or [span]


def _sentences(text: str) -> list[tuple[str, int]]:
    """Sentence-ish pieces with their offsets, preserving every character."""
    out: list[tuple[str, int]] = []
    position = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        out.append((text[position:end], position))
        position = end
    if position < len(text):
        out.append((text[position:], position))
    return out or [(text, 0)]


def _subdivide(span: Span, start: int, end: int, text: str) -> Span:
    """A slice of a span, with its locator narrowed to match (§2.1 rule 4)."""
    locator = span.locator
    if isinstance(locator, TimeLocator):
        narrowed: Locator = locator.subdivide(start, end, length=len(span.text))
    else:
        narrowed = locator.subdivide(start, end)
    return Span(text=text, locator=narrowed, kind=span.kind, meta=dict(span.meta))


def _merge_locators(locators: list[Locator]) -> Locator:
    """One locator covering several spans, when they can honestly be covered.

    Where a union would be a guess — two different pages, two different sheets —
    the first locator is kept rather than inventing a range that spans them. An
    over-broad citation is a citation that sends somebody to the wrong place.
    """
    first = locators[0]
    if len(locators) == 1:
        return first

    if isinstance(first, FileLocator) and all(isinstance(loc, FileLocator) for loc in locators):
        pages = {loc.page for loc in locators}  # type: ignore[union-attr]
        if len(pages) == 1:
            return FileLocator(
                char_start=min(loc.char_start for loc in locators),  # type: ignore[union-attr]
                char_end=max(loc.char_end for loc in locators),  # type: ignore[union-attr]
                page=first.page,
            )
        return first

    if isinstance(first, TimeLocator) and all(isinstance(loc, TimeLocator) for loc in locators):
        channels = {loc.channel for loc in locators}  # type: ignore[union-attr]
        speakers = {loc.speaker for loc in locators}  # type: ignore[union-attr]
        return TimeLocator(
            start_sec=min(loc.start_sec for loc in locators),  # type: ignore[union-attr]
            end_sec=max(loc.end_sec for loc in locators),  # type: ignore[union-attr]
            # Only claim a channel or speaker when every turn agrees; a chunk
            # spanning two speakers belongs to neither.
            channel=first.channel if len(channels) == 1 else None,
            speaker=first.speaker if len(speakers) == 1 else None,
        )

    if isinstance(first, CellLocator) and all(isinstance(loc, CellLocator) for loc in locators):
        sheets = {loc.sheet for loc in locators}  # type: ignore[union-attr]
        if len(sheets) == 1:
            start = first.cell_range.split(":")[0]
            end = locators[-1].cell_range.split(":")[-1]  # type: ignore[union-attr]
            return CellLocator(sheet=first.sheet, cell_range=f"{start}:{end}")
        return first

    return first


def _build_chunk(
    group: list[Span],
    source: Source,
    config: ChunkingConfig,
    counter: TokenCounter,
    ingested_at: datetime,
    index: int,
) -> Chunk:
    text = "\n".join(span.text for span in group).strip()
    locator = _merge_locators([span.locator for span in group])

    # Derived from the source and the position, not from the text: a chunk whose
    # wording changed is still the same chunk of the same file, and an id that
    # moved would orphan the old row instead of replacing it on re-ingest.
    chunk_id = hashlib.sha256(
        f"{source.source_id}\x00{source.content_hash}\x00{index}".encode()
    ).hexdigest()[:32]

    meta: dict[str, Any] = {}
    for span in group:
        meta.update(span.meta)

    provenance = ProvenanceRecord(
        source_id=source.source_id,
        source_type=source.source_type,
        source_uri=source.source_uri,
        source_title=source.source_title,
        ingested_at=ingested_at,
        content_hash=source.content_hash,
        locator=locator,
        parent_id=config.parent_id,
        extra=meta,
    )
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        provenance=provenance,
        token_count=counter.count(text),
    )
