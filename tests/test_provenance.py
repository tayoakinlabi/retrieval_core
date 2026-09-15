"""The frozen schema (contract §2).

These tests are stricter than they look. The schema is frozen because indexes
already written to users' disks hold it in place — a change forces a full
reindex on their hardware, which cannot be patched remotely. So the wire format
is asserted explicitly rather than left to whatever the dataclasses happen to
serialise to, and the round-trip is asserted in both directions.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from retrieval_core.provenance import (
    CellLocator,
    Channel,
    FileLocator,
    ProvenanceRecord,
    SourceType,
    StepLocator,
    TimeLocator,
    locator_from_dict,
    locator_to_dict,
)


class TestRendering:
    """§2.1 rule 1: the library renders; the product resolves."""

    def test_a_page_is_what_a_reader_wants(self):
        assert FileLocator(0, 100, page=14).render() == "p. 14"

    def test_without_pages_it_falls_back_to_offsets(self):
        # Plain text and HTML have no pages. Offsets are poor but true.
        assert FileLocator(20, 80).render() == "chars 20–80"

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            (723.0, 767.0, "12:03–12:47"),
            (0.0, 5.0, "0:00–0:05"),
            (3723.0, 3780.0, "1:02:03–1:03:00"),  # hours appear only when needed
        ],
    )
    def test_time_reads_like_a_timestamp(self, start, end, expected):
        assert TimeLocator(start, end).render() == expected

    def test_a_step_is_numbered(self):
        assert StepLocator(4, "Open the valve").render() == "Step 4"

    def test_a_cell_range_is_sheet_qualified(self):
        assert CellLocator("Q3 Invoices", "B2:B18").render() == "Q3 Invoices!B2:B18"


class TestSubdivision:
    """§2.1 rule 4: a chunk's locator points at the chunk, not its whole span."""

    def test_a_file_slice_is_offset_from_the_span_start(self):
        narrowed = FileLocator(1000, 2000, page=3).subdivide(50, 150)
        assert (narrowed.char_start, narrowed.char_end) == (1050, 1150)
        assert narrowed.page == 3

    def test_a_time_slice_interpolates_by_position(self):
        # 100 characters spanning 10 seconds; the middle fifth is ~4s–6s.
        narrowed = TimeLocator(10.0, 20.0).subdivide(40, 60, length=100)
        assert narrowed.start_sec == pytest.approx(14.0)
        assert narrowed.end_sec == pytest.approx(16.0)

    def test_a_time_slice_keeps_who_was_speaking(self):
        narrowed = TimeLocator(0.0, 10.0, channel=Channel.MICROPHONE, speaker="Ada").subdivide(
            0, 5, length=10
        )
        assert narrowed.channel is Channel.MICROPHONE
        assert narrowed.speaker == "Ada"

    def test_a_zero_length_span_does_not_divide_by_zero(self):
        locator = TimeLocator(1.0, 2.0)
        assert locator.subdivide(0, 0, length=0) == locator

    def test_a_step_stays_the_step_it_was(self):
        # There is no unit below a step that a person following instructions
        # could act on, so a hard-split step still cites the step.
        locator = StepLocator(7)
        assert locator.subdivide(0, 10) == locator


class TestSerialisation:
    @pytest.mark.parametrize(
        "locator",
        [
            FileLocator(10, 20, page=2),
            FileLocator(10, 20),
            TimeLocator(1.5, 9.25, channel=Channel.SYSTEM, speaker="Bob"),
            TimeLocator(0.0, 1.0),
            StepLocator(3, "Attach the hose"),
            CellLocator("Sheet1", "A1:C9"),
        ],
    )
    def test_every_variant_round_trips(self, locator):
        assert locator_from_dict(locator_to_dict(locator)) == locator

    def test_the_kind_tag_is_part_of_the_wire_format(self):
        # Frozen names, written out rather than derived from class names, so a
        # later rename cannot silently invalidate every index on disk.
        assert locator_to_dict(FileLocator(0, 1))["kind"] == "file"
        assert locator_to_dict(TimeLocator(0, 1))["kind"] == "time"
        assert locator_to_dict(StepLocator(1))["kind"] == "step"
        assert locator_to_dict(CellLocator("s", "A1"))["kind"] == "cell"

    def test_an_unknown_kind_is_refused_rather_than_guessed(self):
        """§2.1 rule 3 means variants are only ever *added*.

        So a kind this build does not recognise is an index written by a newer
        one. Degrading to a partial record would put a wrong citation in front
        of somebody, which is worse than refusing to read it.
        """
        with pytest.raises(ValueError, match="newer build"):
            locator_from_dict({"kind": "quantum", "whatever": 1})

    def test_a_record_round_trips_whole(self):
        record = ProvenanceRecord(
            source_id="src-1",
            source_type=SourceType.RECORDING,
            source_uri="C:/audio/call.m4a",
            source_title="Board call.m4a",
            ingested_at=datetime(2026, 9, 15, 12, 0, tzinfo=UTC),
            content_hash="sha256:abc",
            locator=TimeLocator(723.0, 767.0, speaker="Ada"),
            parent_id="parent-9",
            extra={"meeting": "board"},
        )
        assert ProvenanceRecord.from_json(record.to_json()) == record

    def test_extra_survives_untouched(self):
        # The library stores it and hands it back; anything the library needs to
        # understand belongs in a field of its own, where the freeze applies.
        record = ProvenanceRecord(
            source_id="s",
            source_type=SourceType.FILE,
            source_uri="u",
            source_title="t",
            ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
            content_hash="h",
            locator=FileLocator(0, 1),
            extra={"nested": {"anything": [1, 2, 3]}},
        )
        assert ProvenanceRecord.from_json(record.to_json()).extra == {
            "nested": {"anything": [1, 2, 3]}
        }


class TestCitation:
    def test_a_citation_names_the_source_and_the_place(self):
        record = ProvenanceRecord(
            source_id="s",
            source_type=SourceType.FILE,
            source_uri="C:/Shared/Q3.pdf",
            source_title="Q3.pdf",
            ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
            content_hash="h",
            locator=FileLocator(0, 100, page=14),
        )
        assert record.cite() == "Q3.pdf, p. 14"
