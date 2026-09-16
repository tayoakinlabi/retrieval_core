"""Storage, and the deletion cascade the contract calls REQUIRED (§4.1).

Most of this file is about one method. `delete_by_source` is the single worst
place in the portfolio for a bug, because a purge that reports success while
leaving text searchable does not fail loudly — it converts a privacy feature
into a false assurance, and nobody finds out.

So the test the contract actually asks for is here: **grep the raw database file
for a string that was deleted.** Not "does a query still return it" — a query
would not see free pages or FTS5's shadow tables, which is exactly where deleted
text survives.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from retrieval_core.chunking import Source, chunk_spans
from retrieval_core.provenance import FileLocator, SourceType
from retrieval_core.spans import Span
from retrieval_core.storage import Collection, IndexMismatchError

WHEN = datetime(2026, 9, 15, tzinfo=UTC)
DIM = 8

# A string that appears in no other fixture, so finding it in the file means it
# survived the purge rather than arriving from somewhere else.
SECRET = "zarquon-severance-payout-4417"


def source(source_id: str, title: str = "doc.pdf") -> Source:
    return Source(
        source_id=source_id,
        source_type=SourceType.FILE,
        source_uri=f"C:/docs/{title}",
        source_title=title,
        content_hash=f"hash-{source_id}",
    )


def chunks_for(source_id: str, texts: list[str]):
    spans = []
    position = 0
    for text in texts:
        spans.append(Span(text=text, locator=FileLocator(position, position + len(text), page=1)))
        position += len(text)
    return chunk_spans(
        spans,
        source(source_id),
        strategy="steps",  # one chunk per span, so the test controls the count
        ingested_at=WHEN,
    )


def vectors_for(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((count, DIM), dtype=np.float32)


@pytest.fixture
def collection(tmp_path: Path):
    with Collection.open(tmp_path / "coll", embedding_model="test-model", dimension=DIM) as coll:
        yield coll


class TestACollectionIsADirectory:
    """§5 requirements 1, 2 and 5."""

    def test_it_creates_the_three_files(self, tmp_path: Path):
        path = tmp_path / "coll"
        with Collection.open(path, embedding_model="m", dimension=DIM) as coll:
            coll.upsert(chunks_for("s1", ["hello"]), vectors_for(1))
        assert (path / "meta.json").exists()
        assert (path / "index.sqlite").exists()
        assert (path / "vectors.f32").exists()

    def test_no_absolute_path_is_written_inside(self, tmp_path: Path):
        # Otherwise moving or copying the collection silently breaks it.
        path = tmp_path / "coll"
        with Collection.open(path, embedding_model="m", dimension=DIM) as coll:
            coll.upsert(chunks_for("s1", ["hello"]), vectors_for(1))
        meta = (path / "meta.json").read_text(encoding="utf-8")
        assert str(tmp_path) not in meta
        assert "coll" not in meta

    def test_it_survives_being_moved(self, tmp_path: Path):
        original = tmp_path / "before"
        with Collection.open(original, embedding_model="m", dimension=DIM) as coll:
            coll.upsert(chunks_for("s1", ["movable content"]), vectors_for(1))

        moved = tmp_path / "after"
        original.rename(moved)

        with Collection.open(moved, embedding_model="m", dimension=DIM) as coll:
            assert coll.stats().chunk_count == 1
            assert coll.search_lexical("movable")

    def test_the_version_is_stamped(self, collection):
        assert collection.stats().schema_version == 1


class TestTheModelIsFrozenPerCollection:
    """§6: changing the model means every vector was made by a different function."""

    def test_a_different_model_is_refused(self, tmp_path: Path):
        path = tmp_path / "coll"
        Collection.open(path, embedding_model="nomic-v1.5", dimension=DIM).close()
        with pytest.raises(IndexMismatchError, match="reindex"):
            Collection.open(path, embedding_model="minilm", dimension=DIM)

    def test_a_different_dimension_is_refused(self, tmp_path: Path):
        path = tmp_path / "coll"
        Collection.open(path, embedding_model="m", dimension=768).close()
        with pytest.raises(IndexMismatchError, match="dimensional"):
            Collection.open(path, embedding_model="m", dimension=256)

    def test_opening_without_naming_a_model_is_allowed(self, tmp_path: Path):
        # A product reading stats or listing sources should not have to know.
        path = tmp_path / "coll"
        Collection.open(path, embedding_model="m", dimension=DIM).close()
        with Collection.open(path) as coll:
            assert coll.stats().embedding_model == "m"


class TestUpsert:
    def test_chunks_and_vectors_land_together(self, collection):
        chunks = chunks_for("s1", ["alpha text", "beta text", "gamma text"])
        collection.upsert(chunks, vectors_for(3))
        stats = collection.stats()
        assert stats.chunk_count == 3
        assert stats.vector_count == 3

    def test_vectors_are_memory_mapped(self, collection):
        collection.upsert(chunks_for("s1", ["a", "b"]), vectors_for(2))
        matrix = collection.vectors()
        assert matrix.shape == (2, DIM)
        assert isinstance(matrix, np.memmap)

    def test_vectors_round_trip_exactly(self, collection):
        original = vectors_for(4, seed=7)
        collection.upsert(chunks_for("s1", ["a", "b", "c", "d"]), original)
        np.testing.assert_array_equal(np.asarray(collection.vectors()), original)

    def test_a_wrong_vector_count_is_refused(self, collection):
        with pytest.raises(ValueError, match="2 chunks but 3 vectors"):
            collection.upsert(chunks_for("s1", ["a", "b"]), vectors_for(3))

    def test_a_wrong_dimension_is_refused(self, collection):
        wrong = np.zeros((1, DIM + 5), dtype=np.float32)
        with pytest.raises(ValueError, match="dimensional"):
            collection.upsert(chunks_for("s1", ["a"]), wrong)

    def test_chunks_can_be_stored_without_vectors(self, collection):
        # Lexical-only indexing, before embeddings exist or when they are off.
        collection.upsert(chunks_for("s1", ["a", "b"]))
        assert collection.stats().chunk_count == 2
        assert collection.stats().vector_count == 0

    def test_re_upserting_the_same_chunk_replaces_it(self, collection):
        chunks = chunks_for("s1", ["original"])
        collection.upsert(chunks)
        collection.upsert(chunks)
        assert collection.stats().chunk_count == 1


class TestLexicalSearch:
    def test_it_finds_what_it_should(self, collection):
        chunks = chunks_for(
            "s1", ["the cat sat on the mat", "quarterly revenue rose twelve percent"]
        )
        collection.upsert(chunks)
        hits = collection.search_lexical("revenue")
        assert len(hits) == 1
        assert collection.get(hits[0][0])[0].startswith("quarterly revenue")

    def test_larger_scores_are_better(self, collection):
        collection.upsert(
            chunks_for("s1", ["revenue revenue revenue", "revenue mentioned once", "nothing here"])
        )
        hits = collection.search_lexical("revenue")
        scores = [score for _, score in hits]
        assert scores == sorted(scores, reverse=True)

    def test_operators_in_a_query_are_searched_for_not_executed(self, collection):
        # A user typing "profit AND loss" wants those words, and an unbalanced
        # quote should not raise in their face.
        collection.upsert(chunks_for("s1", ["profit and loss statement"]))
        assert collection.search_lexical("profit AND loss")
        assert collection.search_lexical('unmatched " quote') == []
        assert collection.search_lexical("margin*") == []

    def test_an_empty_query_returns_nothing(self, collection):
        collection.upsert(chunks_for("s1", ["anything"]))
        assert collection.search_lexical("   ") == []


class TestDeletionCascade:
    """§4.1. The most consequential behaviour in the library."""

    def test_the_chunks_go(self, collection):
        collection.upsert(chunks_for("s1", ["a", "b"]), vectors_for(2))
        collection.upsert(chunks_for("s2", ["c"]), vectors_for(1, seed=1))
        assert collection.delete_by_source("s1") == 2
        assert collection.stats().chunk_count == 1

    def test_deleting_an_unknown_source_is_not_an_error(self, collection):
        assert collection.delete_by_source("never-existed") == 0

    def test_the_text_is_gone_from_lexical_search(self, collection):
        collection.upsert(chunks_for("s1", [f"payment of {SECRET} approved"]))
        assert collection.search_lexical("zarquon")
        collection.delete_by_source("s1")
        assert collection.search_lexical("zarquon") == []

    def test_the_text_is_gone_from_the_raw_file(self, collection):
        """The test §4.1 asks for by name, and the one that actually bites.

        A plain DELETE leaves the terms in FTS5's own b-tree and the rows in
        freed SQLite pages, where a query will not find them and anyone reading
        the file will. Checking the bytes is the only way to know.
        """
        collection.upsert(chunks_for("s1", [f"severance agreed: {SECRET}"]))
        assert collection.contains_text(SECRET), "fixture is wrong: the text was never stored"

        collection.delete_by_source("s1")

        assert not collection.contains_text(SECRET), (
            "deleted text is still readable in the database file — "
            "this is the failure mode §4.1 exists to prevent"
        )

    def test_the_vectors_are_compacted_not_orphaned(self, collection):
        """An embedding of deleted text is not anonymous.

        It sits close in vector space to the text that produced it, so leaving
        the row in the memmap leaves a recoverable trace of something the user
        asked to be gone.
        """
        first = vectors_for(2, seed=1)
        second = vectors_for(3, seed=2)
        collection.upsert(chunks_for("s1", ["a", "b"]), first)
        collection.upsert(chunks_for("s2", ["c", "d", "e"]), second)
        assert collection.stats().vector_count == 5

        collection.delete_by_source("s1")

        assert collection.stats().vector_count == 3
        # And the survivors are the right ones, in the right order.
        np.testing.assert_array_equal(np.asarray(collection.vectors()), second)

    def test_surviving_chunks_still_point_at_their_own_vectors(self, collection):
        """Compaction renumbers, and getting that wrong is invisible.

        Every search would return plausible, wrong results with nothing to see.
        """
        keep = vectors_for(3, seed=9)
        collection.upsert(chunks_for("s1", ["doomed one", "doomed two"]), vectors_for(2, seed=8))
        survivors = chunks_for("s2", ["kept one", "kept two", "kept three"])
        collection.upsert(survivors, keep)

        collection.delete_by_source("s1")

        matrix = np.asarray(collection.vectors())
        rows = {
            row["chunk_id"]: row["vector_row"]
            for row in collection._db.execute(
                "SELECT chunk_id, vector_row FROM chunks ORDER BY vector_row"
            )
        }
        for index, chunk in enumerate(survivors):
            np.testing.assert_array_equal(matrix[rows[chunk.chunk_id]], keep[index])

    def test_dangling_parents_are_cleared(self, collection):
        """§4.1 item 4: a child outliving its parent breaks small-to-big."""
        from retrieval_core.chunking import ChunkingConfig

        parent = chunks_for("s1", ["the parent section"])
        child_spans = [Span(text="a child chunk", locator=FileLocator(0, 13, page=1))]
        child = chunk_spans(
            child_spans,
            source("s2"),
            strategy="steps",
            config=ChunkingConfig(parent_id=parent[0].chunk_id),
            ingested_at=WHEN,
        )
        collection.upsert(parent)
        collection.upsert(child)

        collection.delete_by_source("s1")

        _, record = collection.get(child[0].chunk_id)
        assert record.parent_id is None

    def test_verify_deleted_agrees(self, collection):
        collection.upsert(chunks_for("s1", [f"contains {SECRET}"]), vectors_for(1))
        assert not collection.verify_deleted("s1")
        collection.delete_by_source("s1")
        assert collection.verify_deleted("s1")

    def test_deleting_one_source_leaves_the_others_searchable(self, collection):
        collection.upsert(chunks_for("s1", ["alpha unique_alpha"]))
        collection.upsert(chunks_for("s2", ["beta unique_beta"]))
        collection.delete_by_source("s1")
        assert collection.search_lexical("unique_alpha") == []
        assert collection.search_lexical("unique_beta")

    def test_everything_can_be_deleted(self, collection):
        collection.upsert(chunks_for("s1", ["a"]), vectors_for(1))
        collection.delete_by_source("s1")
        assert collection.stats().chunk_count == 0
        assert collection.stats().vector_count == 0
        assert collection.vectors().shape == (0, DIM)


class TestCorruptionIsReportedNotLimpedPast:
    def test_a_truncated_vector_file_is_caught_on_open(self, tmp_path: Path):
        """A crash mid-compaction must not become silently wrong searches."""
        path = tmp_path / "coll"
        with Collection.open(path, embedding_model="m", dimension=DIM) as coll:
            coll.upsert(chunks_for("s1", ["a", "b", "c"]), vectors_for(3))

        vectors = path / "vectors.f32"
        data = vectors.read_bytes()
        vectors.write_bytes(data[: len(data) // 3])  # lose two rows

        with pytest.raises(IndexMismatchError, match="reindex"):
            Collection.open(path, embedding_model="m", dimension=DIM)


class TestSummaries:
    def test_sources_are_listed_with_their_titles(self, collection):
        collection.upsert(chunks_for("s1", ["a", "b"]))
        collection.upsert(chunks_for("s2", ["c"]))
        summaries = {s.source_id: s for s in collection.get_sources()}
        assert summaries["s1"].chunk_count == 2
        assert summaries["s2"].chunk_count == 1
        assert summaries["s1"].source_title == "doc.pdf"
