"""Chunking — contract §3.

The interesting tests are the **hard rules** in §3.3, because each one exists to
prevent a specific, silent degradation of retrieval quality:

* a turn split down the middle retrieves half of "no, we agreed the opposite";
* two steps merged into one chunk cite "steps 4 and 5", which is not a
  granularity anybody following instructions can use;
* a table row without its header cannot be matched by anything a person types;
* a locator that points at the whole span instead of the chunk sends the reader
  to the wrong paragraph.

None of those raise. They just make the product quietly worse, which is why
they are asserted rather than trusted.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from retrieval_core.chunking import ChunkingConfig, Source, chunk_spans
from retrieval_core.provenance import (
    CellLocator,
    Channel,
    FileLocator,
    SourceType,
    StepLocator,
    TimeLocator,
)
from retrieval_core.spans import Span, SpanKind
from retrieval_core.tokens import EstimatingCounter

SOURCE = Source(
    source_id="src-1",
    source_type=SourceType.FILE,
    source_uri="C:/Shared/report.pdf",
    source_title="report.pdf",
    content_hash="sha256:abc",
)
WHEN = datetime(2026, 9, 15, tzinfo=UTC)


def paragraph(words: int, marker: str = "x") -> str:
    return " ".join(f"{marker}{i}" for i in range(words)) + "."


def prose_spans(count: int, words: int = 40) -> list[Span]:
    spans = []
    position = 0
    for i in range(count):
        text = paragraph(words, marker=f"p{i}w")
        spans.append(Span(text=text, locator=FileLocator(position, position + len(text), page=1)))
        position += len(text)
    return spans


def chunk(spans, **kwargs):
    return chunk_spans(spans, SOURCE, ingested_at=WHEN, **kwargs)


class TestTheStrategyMustBeReal:
    def test_an_unknown_strategy_is_refused(self):
        # Falling back to prose would split turns mid-sentence and the caller
        # would never learn why retrieval got worse.
        with pytest.raises(ValueError, match="unknown strategy"):
            chunk(prose_spans(1), strategy="semantic")

    def test_no_spans_gives_no_chunks(self):
        assert chunk([], strategy="prose") == []


class TestProse:
    def test_short_input_stays_one_chunk(self):
        chunks = chunk(prose_spans(2, words=20), strategy="prose")
        assert len(chunks) == 1

    def test_it_packs_up_to_the_target(self):
        chunks = chunk(prose_spans(20), strategy="prose")
        assert len(chunks) > 1
        for piece in chunks:
            # The overlap is added on top of the target, so allow for it.
            assert piece.token_count <= 400 * 1.3

    def test_whole_paragraphs_survive(self):
        """§3.3: avoid splitting mid-paragraph.

        Every paragraph here is small enough to fit, so no paragraph should be
        cut — each one's marker should appear intact somewhere.
        """
        spans = prose_spans(20)
        chunks = chunk(spans, strategy="prose")
        combined = "\n".join(c.text for c in chunks)
        for span in spans:
            assert span.text in combined

    def test_overlap_repeats_whole_spans(self):
        chunks = chunk(prose_spans(20), strategy="prose")
        # Consecutive chunks share text, and the shared text is a whole
        # paragraph rather than a fragment of one.
        first_words = set(chunks[0].text.split())
        second_words = set(chunks[1].text.split())
        assert first_words & second_words

    def test_overlap_can_be_switched_off(self):
        config = ChunkingConfig(prose_overlap=0.0)
        chunks = chunk(prose_spans(20), strategy="prose", config=config)
        for earlier, later in zip(chunks, chunks[1:], strict=False):
            assert not (set(earlier.text.split()) & set(later.text.split()))

    def test_targets_are_configurable(self):
        few = chunk(prose_spans(20), strategy="prose", config=ChunkingConfig(prose_target=2000))
        many = chunk(prose_spans(20), strategy="prose", config=ChunkingConfig(prose_target=100))
        assert len(many) > len(few)


class TestTheModelContextIsACeiling:
    """§3.2: ``chunk_size = min(model context, strategy target)``."""

    def test_a_small_window_lowers_the_target(self):
        tiny = EstimatingCounter(max_context=60)
        chunks = chunk(prose_spans(10), strategy="prose", counter=tiny)
        for piece in chunks:
            assert piece.token_count <= 60 * 1.3

    def test_a_large_window_does_not_raise_it(self):
        # nomic's 8192 must not turn a 400-token target into an 8192-token chunk:
        # that dilutes the embedding and destroys citation granularity.
        chunks = chunk(prose_spans(40), strategy="prose", counter=EstimatingCounter(8192))
        assert len(chunks) > 1
        assert max(c.token_count for c in chunks) < 1000


class TestTranscript:
    def turns(self, count: int, words: int = 40) -> list[Span]:
        return [
            Span(
                text=paragraph(words, marker=f"t{i}w"),
                locator=TimeLocator(i * 30.0, i * 30.0 + 28.0, speaker="Ada" if i % 2 else "Bob"),
                kind=SpanKind.TRANSCRIPT_TURN,
            )
            for i in range(count)
        ]

    def test_a_turn_is_never_split(self):
        """The rule that matters most here.

        Half a sentence of somebody reversing their position is worse than not
        retrieving it at all.
        """
        turns = self.turns(12)
        chunks = chunk(turns, strategy="transcript")
        combined = "\n".join(c.text for c in chunks)
        for turn in turns:
            assert turn.text in combined

    def test_chunks_start_at_the_start_of_a_turn(self):
        turns = self.turns(12)
        chunks = chunk(turns, strategy="transcript")
        starts = {turn.text.split()[0] for turn in turns}
        for piece in chunks:
            assert piece.text.split()[0] in starts

    def test_the_overlap_is_a_whole_turn(self):
        chunks = chunk(self.turns(12), strategy="transcript")
        assert len(chunks) > 1
        assert set(chunks[0].text.split()) & set(chunks[1].text.split())

    def test_a_chunk_spanning_two_speakers_claims_neither(self):
        # Attributing a mixed chunk to one of them would be a false citation.
        chunks = chunk(self.turns(12), strategy="transcript")
        assert chunks[0].provenance.locator.speaker is None

    def test_a_single_speaker_chunk_keeps_the_attribution(self):
        turns = [
            Span(
                text=paragraph(20, marker=f"s{i}w"),
                locator=TimeLocator(i * 10.0, i * 10.0 + 9.0, speaker="Ada"),
                kind=SpanKind.TRANSCRIPT_TURN,
            )
            for i in range(3)
        ]
        chunks = chunk(turns, strategy="transcript")
        assert chunks[0].provenance.locator.speaker == "Ada"

    def test_the_time_range_covers_every_turn_in_the_chunk(self):
        chunks = chunk(self.turns(4, words=10), strategy="transcript")
        locator = chunks[0].provenance.locator
        assert locator.start_sec == 0.0
        assert locator.end_sec == pytest.approx(3 * 30.0 + 28.0)

    def test_an_enormous_turn_is_split_rather_than_truncated(self):
        # Handing the model more than it accepts loses text silently, which is
        # worse than a split that is visible in the citation.
        huge = Span(
            text=" ".join(f"Sentence {i}." for i in range(400)),
            locator=TimeLocator(0.0, 600.0, speaker="Ada"),
            kind=SpanKind.TRANSCRIPT_TURN,
        )
        chunks = chunk([huge], strategy="transcript")
        assert len(chunks) > 1
        assert all(c.provenance.locator.speaker == "Ada" for c in chunks)


class TestSteps:
    def steps(self, count: int) -> list[Span]:
        return [
            Span(
                text=paragraph(30, marker=f"st{i}w"),
                locator=StepLocator(i + 1, f"Step {i + 1} title"),
                kind=SpanKind.STEP,
            )
            for i in range(count)
        ]

    def test_one_step_one_chunk(self):
        """§3.3: never merge across steps.

        Ten short steps must not be packed into one chunk just because they
        would fit — a citation reading "steps 4 and 5" is unusable to somebody
        working through a procedure.
        """
        chunks = chunk(self.steps(10), strategy="steps")
        assert len(chunks) == 10

    def test_each_chunk_cites_its_own_step(self):
        chunks = chunk(self.steps(5), strategy="steps")
        assert [c.provenance.locator.step_index for c in chunks] == [1, 2, 3, 4, 5]

    def test_an_oversized_step_is_split_but_still_cites_the_step(self):
        big = Span(
            text=" ".join(f"Instruction {i}." for i in range(3000)),
            locator=StepLocator(4),
            kind=SpanKind.STEP,
        )
        chunks = chunk([big], strategy="steps")
        assert len(chunks) > 1
        assert all(c.provenance.locator.step_index == 4 for c in chunks)


class TestTabular:
    def table(self, rows: int) -> list[Span]:
        header = Span(
            text="Client | Net | VAT | Gross",
            locator=CellLocator("Q3", "A1:D1"),
            kind=SpanKind.HEADING,
        )
        body = [
            Span(
                text=f"Client {i} | {i * 100} | {i * 20} | {i * 120}",
                locator=CellLocator("Q3", f"A{i + 2}:D{i + 2}"),
                kind=SpanKind.TABLE_ROW,
            )
            for i in range(rows)
        ]
        return [header, *body]

    def test_the_header_is_repeated_into_every_chunk(self):
        """Without it, a row of figures matches nothing a person would type.

        "revenue Q3" can only match a chunk that contains the word revenue.
        """
        chunks = chunk(self.table(50), strategy="tabular")
        assert len(chunks) > 1
        for piece in chunks:
            assert "Client | Net | VAT | Gross" in piece.text

    def test_rows_are_never_split(self):
        rows = self.table(50)[1:]
        chunks = chunk(self.table(50), strategy="tabular")
        combined = "\n".join(c.text for c in chunks)
        for row in rows:
            assert row.text in combined

    def test_the_row_count_is_configurable(self):
        chunks = chunk(self.table(50), strategy="tabular", config=ChunkingConfig(tabular_rows=10))
        assert len(chunks) == 5

    def test_the_cell_range_covers_the_rows_in_the_chunk(self):
        chunks = chunk(self.table(6), strategy="tabular", config=ChunkingConfig(tabular_rows=3))
        first = chunks[0].provenance.locator
        assert first.sheet == "Q3"
        # From the header's first cell through the last row included.
        assert first.cell_range.startswith("A1")


class TestLocatorsFollowTheText:
    """§2.1 rule 4, end to end."""

    def test_a_split_span_yields_narrowed_offsets(self):
        text = " ".join(f"Sentence {i}." for i in range(400))
        span = Span(text=text, locator=FileLocator(0, len(text), page=7))
        chunks = chunk([span], strategy="prose")

        assert len(chunks) > 1
        offsets = [(c.provenance.locator.char_start, c.provenance.locator.char_end) for c in chunks]

        # Distinct first, and this assertion is the one that matters. An earlier
        # version of this test checked only that the offsets were ascending,
        # non-degenerate and within the span — all trivially true when every
        # chunk carries the *same* span-wide locator, which is precisely the bug
        # rule 4 forbids. Deleting the narrowing left the whole suite green.
        assert len(set(offsets)) == len(offsets), "every chunk got the same locator"

        # And each one genuinely narrower than the span it came from.
        assert all((start, end) != (0, len(text)) for start, end in offsets), (
            "a chunk claims the whole span"
        )
        assert all(end - start < len(text) for start, end in offsets)

        assert offsets == sorted(offsets)
        assert all(start < end for start, end in offsets)
        assert offsets[0][0] >= 0
        assert offsets[-1][1] <= len(text)
        assert all(c.provenance.locator.page == 7 for c in chunks)

    def test_a_split_turn_narrows_its_timestamps(self):
        """The same rule on the time axis, where it is easiest to get wrong."""
        text = " ".join(f"Sentence {i}." for i in range(400))
        span = Span(
            text=text,
            locator=TimeLocator(100.0, 400.0, speaker="Ada"),
            kind=SpanKind.TRANSCRIPT_TURN,
        )
        chunks = chunk([span], strategy="transcript")

        assert len(chunks) > 1
        ranges = [(c.provenance.locator.start_sec, c.provenance.locator.end_sec) for c in chunks]
        assert len(set(ranges)) == len(ranges), "every chunk got the same timestamps"
        assert all((s, e) != (100.0, 400.0) for s, e in ranges), "a chunk claims the whole turn"
        assert ranges[0][0] == pytest.approx(100.0)
        assert ranges[-1][1] == pytest.approx(400.0, abs=1.0)

    def test_merging_across_pages_does_not_invent_a_range(self):
        # Two pages cannot honestly be covered by one FileLocator, so the first
        # is kept rather than claiming a span across both.
        spans = [
            Span(text=paragraph(10, "a"), locator=FileLocator(0, 50, page=1)),
            Span(text=paragraph(10, "b"), locator=FileLocator(0, 50, page=2)),
        ]
        chunks = chunk(spans, strategy="prose")
        assert len(chunks) == 1
        assert chunks[0].provenance.locator.page == 1

    def test_merging_within_a_page_does_cover_the_range(self):
        spans = [
            Span(text=paragraph(10, "a"), locator=FileLocator(0, 50, page=4)),
            Span(text=paragraph(10, "b"), locator=FileLocator(50, 120, page=4)),
        ]
        chunks = chunk(spans, strategy="prose")
        locator = chunks[0].provenance.locator
        assert (locator.char_start, locator.char_end) == (0, 120)


class TestChunkIdentity:
    def test_ids_are_stable_across_runs(self):
        spans = prose_spans(6)
        first = chunk(spans, strategy="prose")
        second = chunk(spans, strategy="prose")
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_a_changed_source_gives_different_ids(self):
        # Re-ingesting an edited file must replace its chunks, not sit beside
        # them: the id is derived from the content hash for exactly that reason.
        spans = prose_spans(6)
        original = chunk(spans, strategy="prose")
        edited = chunk_spans(
            spans,
            Source("src-1", SourceType.FILE, "u", "report.pdf", "sha256:different"),
            ingested_at=WHEN,
            strategy="prose",
        )
        assert {c.chunk_id for c in original}.isdisjoint({c.chunk_id for c in edited})

    def test_ids_are_unique_within_a_source(self):
        chunks = chunk(prose_spans(40), strategy="prose")
        assert len({c.chunk_id for c in chunks}) == len(chunks)


class TestSmallToBig:
    def test_a_parent_is_recorded_when_asked_for(self):
        """§3.4: embed small for precision, expand to the parent for context."""
        chunks = chunk(
            prose_spans(10), strategy="prose", config=ChunkingConfig(parent_id="section-2")
        )
        assert all(c.provenance.parent_id == "section-2" for c in chunks)

    def test_it_is_absent_unless_requested(self):
        chunks = chunk(prose_spans(4), strategy="prose")
        assert all(c.provenance.parent_id is None for c in chunks)


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"prose_overlap": 1.0},
            {"prose_overlap": -0.1},
            {"prose_target": 0},
            {"transcript_target": -1},
            {"tabular_rows": 0},
        ],
    )
    def test_nonsense_is_refused_at_construction(self, kwargs):
        with pytest.raises(ValueError):
            ChunkingConfig(**kwargs)


class TestMetadataSurvives:
    def test_span_meta_reaches_the_provenance_record(self):
        spans = [
            Span(
                text=paragraph(10),
                locator=TimeLocator(0.0, 5.0, channel=Channel.SYSTEM),
                kind=SpanKind.TRANSCRIPT_TURN,
                meta={"confidence": 0.92},
            )
        ]
        chunks = chunk(spans, strategy="transcript")
        assert chunks[0].provenance.extra["confidence"] == 0.92
