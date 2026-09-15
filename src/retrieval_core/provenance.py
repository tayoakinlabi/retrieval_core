"""Where a chunk came from. **This schema is frozen** (contract §2).

Everything else in this library can be rewritten freely — §4's API is explicitly
not a compatibility promise, because the only consumers are three products in
the same hands, pinned by commit. This module is the exception, and the reason
is not taste:

    Once somebody has indexed thousands of documents on their own machine, a
    schema change forces a full reindex on their hardware, and you cannot patch
    that remotely.

So a locator is a **tagged union, not an offset**. A bare `char_start` cannot
say "page 14" or "12:03–12:47", and a citation nobody can act on is the feature
failing quietly. Rule 3 of §2.1 follows from the same place: adding a variant
later must never migrate existing records, which is why each variant is its own
type carrying its own fields rather than a widening bag of optional ones.

Two responsibilities are split deliberately (§2.1 rules 1 and 2):

* the **library** renders a locator to something a person reads — ``p. 14``,
  ``12:03–12:47``, ``Step 4``;
* the **product** turns it into a jump, because only the product knows whether
  that means scrolling a PDF, seeking a recording, or selecting a cell range.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

__all__ = [
    "CellLocator",
    "FileLocator",
    "Locator",
    "ProvenanceRecord",
    "SourceType",
    "StepLocator",
    "TimeLocator",
    "locator_from_dict",
]


class SourceType(StrEnum):
    FILE = "file"
    RECORDING = "recording"
    PROCEDURE = "procedure"
    MANUAL = "manual"


class Channel(StrEnum):
    """Which side of a recording a span came from (Minutebook M2).

    Captured as two channels rather than one mix: "the client said it" and "we
    said it" are different facts, and a decision ledger that cannot tell them
    apart is worth much less.
    """

    MICROPHONE = "microphone"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class FileLocator:
    """A place in a document. ``page`` is None for formats without pages."""

    char_start: int
    char_end: int
    page: int | None = None

    def render(self) -> str:
        if self.page is not None:
            return f"p. {self.page}"
        return f"chars {self.char_start}–{self.char_end}"

    def subdivide(self, start: int, end: int) -> FileLocator:
        """Narrow to a slice of this span, offsets relative to the span's text."""
        return FileLocator(
            char_start=self.char_start + start,
            char_end=self.char_start + end,
            page=self.page,
        )


@dataclass(frozen=True, slots=True)
class TimeLocator:
    """A stretch of a recording."""

    start_sec: float
    end_sec: float
    channel: Channel | None = None
    speaker: str | None = None

    def render(self) -> str:
        return f"{_timestamp(self.start_sec)}–{_timestamp(self.end_sec)}"

    def subdivide(self, start: int, end: int, *, length: int) -> TimeLocator:
        """Narrow proportionally, by how far into the text the slice sits.

        An approximation, and knowingly so: the library is handed a turn with
        one start and one end, not per-word timings. Interpolating by character
        position puts a citation within a second or two of the right place in
        ordinary speech, which is what someone scrubbing a recording needs. A
        parser that *does* have word timings should hand in finer spans instead
        of relying on this.
        """
        if length <= 0:
            return self
        duration = self.end_sec - self.start_sec
        return TimeLocator(
            start_sec=self.start_sec + duration * (start / length),
            end_sec=self.start_sec + duration * (end / length),
            channel=self.channel,
            speaker=self.speaker,
        )


@dataclass(frozen=True, slots=True)
class StepLocator:
    """One step of a procedure. Steps are never merged or split (§3.3)."""

    step_index: int
    step_title: str | None = None

    def render(self) -> str:
        return f"Step {self.step_index}"

    def subdivide(self, start: int, end: int) -> StepLocator:
        # A step that had to be hard-split is still that step; there is no
        # smaller unit a reader could act on.
        return self


@dataclass(frozen=True, slots=True)
class CellLocator:
    """A range in a spreadsheet."""

    sheet: str
    cell_range: str

    def render(self) -> str:
        return f"{self.sheet}!{self.cell_range}"

    def subdivide(self, start: int, end: int) -> CellLocator:
        # Narrowing would mean recomputing a range from row counts, which the
        # library cannot do without knowing the table's shape. The tabular
        # strategy never splits a row, so the range stays truthful.
        return self


Locator = FileLocator | TimeLocator | StepLocator | CellLocator

# Wire names, stored in the database. Changing one of these is the migration the
# freeze exists to prevent, so they are written out rather than derived from
# class names that somebody might later rename.
_LOCATOR_KIND = {
    FileLocator: "file",
    TimeLocator: "time",
    StepLocator: "step",
    CellLocator: "cell",
}
_LOCATOR_TYPE = {name: cls for cls, name in _LOCATOR_KIND.items()}


def _timestamp(seconds: float) -> str:
    """``12:03``, or ``1:02:03`` once it runs past an hour."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def locator_to_dict(locator: Locator) -> dict[str, Any]:
    """Serialise, tagged with its kind so it can be read back as itself."""
    kind = _LOCATOR_KIND.get(type(locator))
    if kind is None:
        raise TypeError(f"not a locator: {type(locator).__name__}")
    fields = {f: getattr(locator, f) for f in locator.__slots__}
    return {"kind": kind, **{k: _plain(v) for k, v in fields.items()}}


def _plain(value: Any) -> Any:
    return value.value if isinstance(value, StrEnum) else value


def locator_from_dict(data: dict[str, Any]) -> Locator:
    """Read a locator back.

    An unknown kind raises rather than degrading to a partial record. §2.1 rule
    3 means new variants are *added*, so a kind this build does not recognise is
    an index written by a newer version — and guessing at it would put a wrong
    citation in front of somebody.
    """
    kind = data.get("kind")
    cls = _LOCATOR_TYPE.get(kind)
    if cls is None:
        raise ValueError(
            f"unknown locator kind {kind!r}; this index was probably written by a newer build"
        )
    fields = {k: v for k, v in data.items() if k != "kind"}
    if cls is TimeLocator and fields.get("channel") is not None:
        fields["channel"] = Channel(fields["channel"])
    return cls(**fields)


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Everything needed to cite one chunk, and to find it again later."""

    source_id: str
    source_type: SourceType
    source_uri: str
    source_title: str
    ingested_at: datetime
    content_hash: str
    locator: Locator
    parent_id: str | None = None
    # Product-specific and deliberately uninterpreted: the library stores it and
    # hands it back. Anything the library needs to understand belongs in a field
    # of its own, where the freeze applies to it.
    extra: dict[str, Any] = field(default_factory=dict)

    def cite(self) -> str:
        """A human-readable citation: ``Accounts.pdf, p. 14``."""
        return f"{self.source_title}, {self.locator.render()}"

    def to_json(self) -> str:
        return json.dumps(
            {
                "source_id": self.source_id,
                "source_type": self.source_type.value,
                "source_uri": self.source_uri,
                "source_title": self.source_title,
                "ingested_at": self.ingested_at.isoformat(),
                "content_hash": self.content_hash,
                "locator": locator_to_dict(self.locator),
                "parent_id": self.parent_id,
                "extra": self.extra,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> ProvenanceRecord:
        data = json.loads(raw)
        return cls(
            source_id=data["source_id"],
            source_type=SourceType(data["source_type"]),
            source_uri=data["source_uri"],
            source_title=data["source_title"],
            ingested_at=datetime.fromisoformat(data["ingested_at"]),
            content_hash=data["content_hash"],
            locator=locator_from_dict(data["locator"]),
            parent_id=data.get("parent_id"),
            extra=data.get("extra") or {},
        )
